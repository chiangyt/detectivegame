"""
Flask web server for the Detective Game.
Bridges the HTML frontend with existing Python backend modules.
"""

import json
import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from script_manager import ScriptManager
from npc_agent_manager import NPCAgentManager
from evidence_handler import EvidenceHandler

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "script.json")

load_dotenv(os.path.join(HERE, "api_key.env"))
_api_key = os.environ.get("OPENAI_API_KEY", "")
_openai_client = openai.OpenAI(api_key=_api_key) if _api_key else None

script_mgr = ScriptManager(SCRIPT_PATH)
raw_data = script_mgr.raw_script

evidence_handler = EvidenceHandler(script_mgr)

agent_manager = NPCAgentManager(
    script_mgr=script_mgr,
    openai_client=_openai_client,
    model="gpt-4o-mini",
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/game-data")
def game_data():
    npcs = [{"npc_name": n["npc_name"]} for n in raw_data["npcs"]]
    return jsonify({
        "case_description": raw_data["case_description"],
        "npcs": npcs,
        "evidence_keys": list(raw_data["evidence"].keys()),
    })


@app.route("/api/ask-npc", methods=["POST"])
def ask_npc():
    data = request.get_json()
    npc_name = data.get("npc_name", "")
    question = data.get("question", "")
    if not npc_name or not question:
        return jsonify({"error": "Missing npc_name or question"}), 400

    def generate():
        for event in agent_manager.ask_stream(npc_name, question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/search-evidence", methods=["POST"])
def search_evidence():
    data = request.get_json()
    item = data.get("item", "")
    if not item:
        return jsonify({"error": "Missing item"}), 400

    result = evidence_handler.search(item)
    matched_key = None
    for key in evidence_handler.evidence:
        nk = evidence_handler.normalize(key)
        nq = evidence_handler.normalize(item)
        if nk == nq or nq in nk or nk in nq:
            matched_key = key
            break
    return jsonify({"result": result, "matched_key": matched_key})


@app.route("/api/conditions")
def get_conditions():
    return jsonify({
        "conditions": script_mgr.get_conditions(),
        "all_met": script_mgr.has_solved_all_conditions(),
    })


@app.route("/api/deduce", methods=["POST"])
def deduce():
    data = request.get_json()
    culprit = data.get("culprit", "")
    location = data.get("location", "")
    motive = data.get("motive", "")
    correct = (
        culprit == "Morgan"
        and location == "glasses"
        and motive == "Economic dispute over contract"
    )
    return jsonify({"correct": correct})


if __name__ == "__main__":
    print(f"[SERVER] Starting Detective Game on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
