"""KG relation schema map, transcript CSV, and interactive HTML visualization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from q2q.kb import is_valid_kg_entity_name
from q2q.schemas import relation_pairs, write_json_atomic


def build_type_subtype_map(kg_data: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    type_to_subtypes: dict[str, set[str]] = defaultdict(set)
    for episode in kg_data:
        episode_knowledge = episode.get("episode_knowledge", {})
        if not isinstance(episode_knowledge, dict):
            continue
        for head, head_info in episode_knowledge.items():
            if not is_valid_kg_entity_name(str(head)) or not isinstance(head_info, dict):
                continue
            relationships = head_info.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for tail, relation_info in relationships.items():
                if not is_valid_kg_entity_name(str(tail)) or not isinstance(relation_info, dict):
                    continue
                relation_type = relation_info.get("type")
                subtype = relation_info.get("subtype")
                relation_types = (
                    [relation_type] if isinstance(relation_type, str)
                    else relation_type if isinstance(relation_type, list) else []
                )
                subtypes = (
                    [subtype] if isinstance(subtype, str)
                    else subtype if isinstance(subtype, list) else []
                )
                if len(relation_types) != len(subtypes):
                    print(
                        f"[kg-expansion] Skipping mismatched type/subtype: "
                        f"{head} -> {tail}",
                        file=sys.stderr,
                    )
                    continue
                for relation_name, subtype_name in zip(relation_types, subtypes):
                    if isinstance(relation_name, str) and isinstance(subtype_name, str):
                        type_to_subtypes[relation_name].add(subtype_name)
    return {
        relation_name: sorted(subtypes)
        for relation_name, subtypes in sorted(type_to_subtypes.items())
    }



def save_transcript_csv(
    path: Path, transcript: Sequence[dict[str, str]], episode_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("level", "episode", "question", "answer")
        )
        writer.writeheader()
        for turn in transcript:
            writer.writerow(
                {
                    "level": "dialogue",
                    "episode": episode_id,
                    "question": turn.get("sub_question", ""),
                    "answer": turn.get("answer", ""),
                }
            )



def kg_graph_signature(
    kg_data: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    nodes: set[str] = set()
    edges: set[str] = set()
    for episode in kg_data:
        episode_id = str(episode.get("episode", ""))
        knowledge = episode.get("episode_knowledge", {})
        if not isinstance(knowledge, dict):
            continue
        for head, info in knowledge.items():
            head_name = str(head)
            nodes.add(head_name)
            if not isinstance(info, dict):
                continue
            relationships = info.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for tail, relation_info in relationships.items():
                tail_name = str(tail)
                nodes.add(tail_name)
                if not isinstance(relation_info, dict):
                    continue
                for relation_type, subtype in relation_pairs(relation_info):
                    edges.add(
                        "|".join(
                            (episode_id, head_name, relation_type, subtype, tail_name)
                        )
                    )
    return {"nodes": sorted(nodes), "edges": sorted(edges)}



def write_kg_visualization_html(
    kg_path: Path,
    output_path: Path,
    baseline_signature: dict[str, list[str]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = os.path.relpath(
        kg_path, start=output_path.parent
    ).replace(os.sep, "/")
    baseline_signature = baseline_signature or {"nodes": [], "edges": []}
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DramaQA KG Comparison</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
:root{--ink:#17211b;--muted:#69746c;--line:#d8d7cc;--green:#174f3d;--orange:#ef6f3c;--paper:#faf9f4}
*{box-sizing:border-box}body{font-family:Inter,Arial,sans-serif;margin:0;color:var(--ink);background:var(--paper)}
#toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:12px 14px;border-bottom:1px solid var(--line);background:white}
#info{font-size:13px;font-weight:700;margin-right:auto}.legend{display:flex;gap:12px;font-size:12px;color:var(--muted)}
.key{display:inline-flex;align-items:center;gap:5px}.line{width:24px;height:3px;border-radius:3px;background:#9ba59e}.line.new{height:5px;background:var(--orange)}
button{border:1px solid var(--line);background:white;border-radius:9px;padding:7px 10px;font-weight:700;color:var(--ink);cursor:pointer}
button.active{background:var(--green);border-color:var(--green);color:white}#network{height:820px;background:white}
</style></head><body>
<div id="toolbar"><div id="info">Loading KG comparison...</div><div class="legend"><span class="key"><i class="line"></i>기존 edge</span><span class="key"><i class="line new"></i>이번에 추가</span></div><button id="showAll" class="active">전체</button><button id="showAdded">추가 edge만</button><button id="relayout">레이아웃 다시 계산</button></div>
<div id="network"></div>
<script>
const KG_URL=__KG_URL__; const baselineNodeIds=new Set(__BASELINE_NODES__); const baselineEdgeIds=new Set(__BASELINE_EDGES__);
const nodes=new vis.DataSet([]); const edges=new vis.DataSet([]);
const network=new vis.Network(document.getElementById("network"),{nodes:nodes,edges:edges},{
  edges:{arrows:{to:{enabled:true}},font:{align:"middle",size:11}},
  nodes:{shape:"dot",font:{size:13}},
  physics:{enabled:true,solver:"forceAtlas2Based",stabilization:{enabled:true,iterations:450}},
  interaction:{hover:true,navigationButtons:true,keyboard:true}
});
let allNodes=[]; let allEdges=[]; let mode="all"; let lastPayload=null;
let freezeTimer=null;
function asArray(value){return Array.isArray(value)?value:(value===undefined||value===null?[]:[value]);}
function collect(data){
  const nodeIds=new Set(); const edgeMap=new Map();
  for(const episode of data){
    const episodeId=String(episode.episode||""); const knowledge=episode.episode_knowledge||{};
    for(const [head,info] of Object.entries(knowledge)){
      nodeIds.add(head);
      for(const [tail,rel] of Object.entries(info.relationships||{})){
        nodeIds.add(tail); const types=asArray(rel.type); const subtypes=asArray(rel.subtype);
        for(let i=0;i<Math.min(types.length,subtypes.length);i++){
          const id=[episodeId,head,types[i],subtypes[i],tail].join("|");
          edgeMap.set(id,{id:id,from:head,to:tail,label:String(subtypes[i]),relationType:String(types[i]),episode:episodeId});
        }
      }
    }
  }
  return {nodeIds:nodeIds,edgeMap:edgeMap};
}
function freezeLayout(){
  clearTimeout(freezeTimer);
  network.setOptions({physics:{enabled:false}});
  network.fit({animation:{duration:250,easingFunction:"easeInOutQuad"}});
}
function settleAndFreeze(){
  clearTimeout(freezeTimer);
  network.setOptions({physics:{enabled:true,solver:"forceAtlas2Based",stabilization:{enabled:true,iterations:450}}});
  network.stabilize(450);
  freezeTimer=setTimeout(freezeLayout,2000);
}
function applyMode(){
  const visibleEdges=mode==="added"?allEdges.filter((edge)=>edge.added):allEdges;
  const connected=new Set(); visibleEdges.forEach((edge)=>{connected.add(edge.from);connected.add(edge.to);});
  const visibleNodes=mode==="added"?allNodes.filter((node)=>connected.has(node.id)):allNodes;
  nodes.clear();edges.clear();nodes.add(visibleNodes);edges.add(visibleEdges);settleAndFreeze();
  document.getElementById("showAll").classList.toggle("active",mode==="all");
  document.getElementById("showAdded").classList.toggle("active",mode==="added");
}
function buildComparison(currentData){
  const current=collect(currentData);
  allNodes=Array.from(current.nodeIds).map((id)=>{
    const added=!baselineNodeIds.has(id); return {id:id,label:id,title:(added?"이번에 추가된 인물: ":"Character: ")+id,color:added?{background:"#ffd7c7",border:"#ef6f3c"}:{background:"#dcebdc",border:"#4f7961"}};
  });
  allEdges=Array.from(current.edgeMap.values()).map((edge)=>{
    const added=!baselineEdgeIds.has(edge.id); return {...edge,added:added,title:(added?"[이번에 추가] ":"[기존] ")+"Episode: "+edge.episode+" | Type: "+edge.relationType,color:{color:added?"#ef6f3c":"#9ba59e",highlight:added?"#d94f1d":"#57645c"},width:added?5:1.2,arrows:"to"};
  });
  const addedCount=allEdges.filter((edge)=>edge.added).length;
  document.getElementById("info").textContent="Nodes: "+allNodes.length+" | 기존 edges: "+baselineEdgeIds.size+" | 현재 edges: "+allEdges.length+" | 이번에 추가: "+addedCount;
  applyMode();
}
async function fetchJson(url){if(!url)return [];const response=await fetch(url+"?timestamp="+Date.now());if(!response.ok)throw new Error("HTTP "+response.status+" for "+url);return response.json();}
async function loadGraph(){
  try{
    const currentData=await fetchJson(KG_URL);
    const payload=JSON.stringify(currentData);if(payload===lastPayload)return;lastPayload=payload;buildComparison(currentData);
  }catch(error){document.getElementById("info").textContent="Error loading KG: "+error.message;}
}
document.getElementById("showAll").onclick=()=>{mode="all";applyMode();};
document.getElementById("showAdded").onclick=()=>{mode="added";applyMode();};
document.getElementById("relayout").onclick=settleAndFreeze;
loadGraph();setInterval(loadGraph,5000);
</script></body></html>
    """.replace("__KG_URL__", json.dumps(relative_path)).replace(
        "__BASELINE_NODES__", json.dumps(baseline_signature["nodes"])
    ).replace(
        "__BASELINE_EDGES__", json.dumps(baseline_signature["edges"])
    )
    output_path.write_text(html, encoding="utf-8")

