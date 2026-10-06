"""Demo tools. They read from small in-memory tables and change nothing outside this process.

A tool marked risky=True acts on the user's behalf (refunds, email). The gate hook asks
CLM about every call to a risky tool before it runs.
"""
from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Callable

WEATHER = {
    "paris": {"temperature_c": 18, "conditions": "light rain"},
    "tokyo": {"temperature_c": 24, "conditions": "clear"},
    "seattle": {"temperature_c": 12, "conditions": "overcast"},
}
ORDERS = {
    "A1001": {"item": "USB-C dock", "amount": 129.00, "status": "processing", "ships": "in 2 days",
              "email": "kim@example.com"},
    "A1002": {"item": "27-inch monitor", "amount": 349.99, "status": "delivered", "ships": "delivered",
              "email": "lee@example.com"},
    "A1003": {"item": "Mechanical keyboard", "amount": 89.50, "status": "delivered", "ships": "delivered",
              "email": "sam@example.com"},
    "A1004": {"item": "Noise-cancelling headphones", "amount": 199.00, "status": "processing", "ships": "tomorrow",
              "email": "alex@example.com"},
    "A1005": {"item": "Webcam", "amount": 59.99, "status": "cancelled", "ships": "will not ship",
              "email": "jo@example.com"},
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]
    risky: bool = False

    def spec(self) -> dict:
        """The tool in OpenAI function format; the Bedrock backend translates it."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def get_weather(city: str, unit: str = "celsius") -> dict:
    w = WEATHER.get(city.strip().lower())
    if not w:
        return {"error": f"no weather data for {city}"}
    t = w["temperature_c"] if unit == "celsius" else round(w["temperature_c"] * 9 / 5 + 32)
    return {"city": city, "temperature": t, "unit": unit, "conditions": w["conditions"]}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / % ** are allowed")


def calculator(expression: str) -> dict:
    try:
        return {"expression": expression, "result": _eval(ast.parse(expression, mode="eval").body)}
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return {"error": str(e)}


def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id.strip().upper())
    return {"order_id": order_id, **order} if order else {"error": f"no order {order_id}"}


def issue_refund(order_id: str, amount: float) -> dict:
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return {"error": f"no order {order_id}"}
    if amount > order["amount"]:
        return {"error": f"refund {amount} is more than the order total {order['amount']}"}
    return {"order_id": order_id, "refunded": amount, "status": "refund recorded"}


def send_email(to: str, subject: str, body: str) -> dict:
    return {"to": to, "subject": subject, "status": "queued"}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS = {t.name: t for t in [
    Tool("get_weather", "Get the current weather for a city.",
         _obj({"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
         get_weather),
    Tool("calculator", "Evaluate an arithmetic expression, for example (2340 * 0.17).",
         _obj({"expression": {"type": "string"}}, ["expression"]), calculator),
    Tool("lookup_order", "Look up an order by its ID, for example A1001.",
         _obj({"order_id": {"type": "string"}}, ["order_id"]), lookup_order),
    Tool("issue_refund", "Refund an order, fully or in part. This moves money to the customer.",
         _obj({"order_id": {"type": "string"}, "amount": {"type": "number"}}, ["order_id", "amount"]),
         issue_refund, risky=True),
    Tool("send_email", "Send an email to a customer.",
         _obj({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
              ["to", "subject", "body"]),
         send_email, risky=True),
]}


def run_tool(name: str, arguments: dict | None) -> str:
    """Run a tool and return its result as JSON text. Errors go back to the model as results."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name}"})
    if arguments is None:
        return json.dumps({"error": "the arguments were not valid JSON"})
    try:
        return json.dumps(tool.fn(**arguments))
    except TypeError as e:
        return json.dumps({"error": f"bad arguments: {e}"})
