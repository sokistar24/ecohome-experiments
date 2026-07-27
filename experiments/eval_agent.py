"""
Experiment agent: three-provider adapter + two action interfaces (G5).

Providers read keys from the project .env / environment:
    openai -> OPENAI_API_KEY, google -> GEMINI_API_KEY,
    anthropic -> ANTHROPIC_KEY (or ANTHROPIC_API_KEY).
Interface "fc"  : LangGraph create_react_agent with native function calling.
Interface "text": a parsed-ReAct loop (Action:/Action Input:/Final Answer:)
                  mirroring the anchor paper's text-action design.
model_key "mock": no LLM at all -- a scripted policy that exercises the
                  tool layer, runner, and scorer offline.
Returns a uniform dict: final, tool_calls (from RunContext), tokens,
snapshot, iterations, error.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from experiments import config
from experiments.eval_tools import RUN

_SNAPSHOT_KEYS = ("model_name", "model", "model_version")


def _make_llm(model_key: str):
    spec = config.MODELS[model_key]
    provider, model = spec["provider"], spec["model"]
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=config.TEMPERATURE,
                          api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "deepinfra":
        # DeepInfra exposes an OpenAI-compatible endpoint, so we reuse the
        # same client with a base_url override. Native function calling is
        # served through the standard tools/tool_calls fields, so the "fc"
        # interface (create_react_agent) works unchanged. response_metadata
        # carries the DeepInfra model string, captured by _snapshot().
        #
        # Some Qwen3 builds default to thinking mode ON, which is documented to
        # plan tool calls in the reasoning trace without emitting them. When a
        # model spec sets enable_thinking=False we forward it through the
        # chat-template kwargs (the vLLM/SGLang/DeepInfra convention) so the
        # served model runs non-thinking and emits clean tool calls. Passed via
        # extra_body so it rides on every request body, not just construction.
        from langchain_openai import ChatOpenAI
        kwargs = dict(model=model, temperature=config.TEMPERATURE,
                      api_key=os.getenv("DEEPINFRA_API_KEY"),
                      base_url="https://api.deepinfra.com/v1/openai",
                      # Cap output tokens: some DeepInfra serves (e.g.
                      # Qwen3-32B, max_model_len=40960) reject the provider
                      # default max_tokens (65536) outright. 8192 is ample for
                      # these scheduling replies and leaves room for the
                      # ~13-18k-token tool-augmented input within the window.
                      max_tokens=config.DEEPINFRA_MAX_TOKENS)
        if "enable_thinking" in spec:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": spec["enable_thinking"]}}
        return ChatOpenAI(**kwargs)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model,
                                      temperature=config.TEMPERATURE,
                                      google_api_key=os.getenv("GEMINI_API_KEY"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model, temperature=config.TEMPERATURE,
                             api_key=key, max_tokens=4096)
    raise ValueError(f"unknown provider {provider}")


def _sum_usage(messages) -> Dict[str, int]:
    tin = tout = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None)
        if u:
            tin += u.get("input_tokens", 0)
            tout += u.get("output_tokens", 0)
    return {"input": tin, "output": tout}


def _snapshot(messages) -> Optional[str]:
    for m in reversed(messages):
        meta = getattr(m, "response_metadata", None) or {}
        for k in _SNAPSHOT_KEYS:
            if meta.get(k):
                return meta[k]
    return None


# ------------------------------------------------------------------ FC
def _invoke_fc(model_key, system_prompt, context, question, toolkit) -> Dict:
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import create_react_agent
    llm = _make_llm(model_key)
    graph = create_react_agent(model=llm, tools=toolkit,
                               prompt=SystemMessage(content=system_prompt))
    msgs = [("system", context), ("user", question)]
    out = graph.invoke({"messages": msgs},
                       config={"recursion_limit": config.AGENT_RECURSION_LIMIT})
    messages = out["messages"]
    ai_iters = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    return {"final": messages[-1].content if messages else "",
            "tokens": _sum_usage(messages),
            "snapshot": _snapshot(messages),
            "iterations": ai_iters, "error": None}


# ---------------------------------------------------------------- text
_TEXT_FORMAT = """

You do not have native tool calling here. To use a tool, reply with EXACTLY:
Action: <tool_name>
Action Input: <JSON object of arguments>
and nothing else. You will then be given the tool result as the next user
message. When completely finished, reply with:
Final Answer: <your summary for the user>
Available tools:
{tool_specs}"""

_ACTION_RE = re.compile(r"Action:\s*([\w_]+)\s*Action Input:\s*(\{.*\})",
                        re.DOTALL)


def _invoke_text(model_key, system_prompt, context, question, toolkit) -> Dict:
    llm = _make_llm(model_key)
    tools = {t.name: t for t in toolkit}
    specs = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}"
                      for t in toolkit)
    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage)
    messages = [SystemMessage(content=system_prompt
                              + _TEXT_FORMAT.format(tool_specs=specs)),
                SystemMessage(content=context),
                HumanMessage(content=question)]
    all_msgs, iters = [], 0
    for _ in range(config.AGENT_RECURSION_LIMIT):
        resp = llm.invoke(messages)
        all_msgs.append(resp)
        iters += 1
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        if "Final Answer:" in text:
            final = text.split("Final Answer:", 1)[1].strip()
            return {"final": final, "tokens": _sum_usage(all_msgs),
                    "snapshot": _snapshot(all_msgs),
                    "iterations": iters, "error": None}
        m = _ACTION_RE.search(text)
        if not m:
            messages += [resp, HumanMessage(content=(
                "Invalid format. Use 'Action:/Action Input:' exactly, or "
                "'Final Answer:' when done."))]
            continue
        name, raw = m.group(1), m.group(2)
        if name not in tools:
            result = {"error": f"unknown tool {name}"}
        else:
            try:
                result = tools[name].invoke(json.loads(raw))
            except Exception as e:
                result = {"error": f"tool raised: {e}"}
        messages += [resp, HumanMessage(content="Tool result: "
                                        + json.dumps(result)[:6000])]
    return {"final": "", "tokens": _sum_usage(all_msgs),
            "snapshot": _snapshot(all_msgs), "iterations": iters,
            "error": "recursion limit reached without Final Answer"}


# ---------------------------------------------------------------- mock
def _invoke_mock(system_prompt, context, question, toolkit) -> Dict:
    """Scripted policy exercising the real tools: prices (retry once on
    error) -> window sums per requested appliance (honouring a 'by HH:MM'
    deadline in the question) -> commit cheapest -> done. Also answers
    date-probe questions from RUN.eval_date. No LLM, zero tokens."""
    if "tomorrow's date" in question.lower():
        from datetime import date as _d, timedelta
        ans = (_d.fromisoformat(RUN.eval_date) + timedelta(days=1)).isoformat()
        return {"final": ans, "tokens": {"input": 0, "output": 0},
                "snapshot": "mock", "iterations": 1, "error": None}
    tools = {t.name: t for t in toolkit}
    first = tools["get_electricity_prices"].invoke({"date": "tomorrow"})
    if isinstance(first, dict) and "error" in first:      # S6: retry once
        tools["get_electricity_prices"].invoke({"date": "tomorrow"})
    wanted = [a for a in config.APPLIANCES
              if a.replace("_", " ") in question.lower()
              or ("ev" in question.lower() and a == "ev_charger")]
    m = re.search(r"by (\d{2}):(\d{2})", question)
    deadline = (int(m.group(1)) * 2 + (1 if int(m.group(2)) >= 30 else 0)
                if m else None)
    for a in wanted or list(config.APPLIANCES):
        slots = config.APPLIANCES[a]["slots"]
        lf = deadline if (a == "ev_charger" and deadline) else None
        res = tools["calculate_window_sums"].invoke(
            {"date": "tomorrow", "window_slots": slots,
             "latest_finish_slot": lf})
        if "error" in res:
            tools["report_infeasibility"].invoke(
                {"explanation": f"{a}: {res['error']}"})
            continue
        best = res["cheapest"][0]["start_slot"]
        tools["schedule_appliance"].invoke(
            {"appliance": a, "start_slot": best, "date": "tomorrow"})
    return {"final": "mock run committed schedules",
            "tokens": {"input": 0, "output": 0}, "snapshot": "mock",
            "iterations": 1, "error": None}


# ---------------------------------------------------------------- api
def invoke_agent(model_key: str, *, interface: str, system_prompt: str,
                 context: str, question: str, toolkit) -> Dict:
    if model_key == "mock":
        return _invoke_mock(system_prompt, context, question, toolkit)
    if interface == "fc":
        return _invoke_fc(model_key, system_prompt, context, question, toolkit)
    if interface == "text":
        return _invoke_text(model_key, system_prompt, context, question, toolkit)
    raise ValueError(f"unknown interface {interface}")
