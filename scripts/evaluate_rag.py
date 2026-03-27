# scripts/evaluate_rag.py
"""
RAG 评测脚本：评估 Deer-Scholar 本地论文检索增强生成系统的质量

================================================================================
使用方法:
================================================================================

1. 基础评测（使用内置示例评测集）:
   uv run python scripts/evaluate_rag.py

2. 使用自定义评测集:
   uv run python scripts/evaluate_rag.py --eval-file path/to/eval_dataset.json

3. 调整检索参数:
   uv run python scripts/evaluate_rag.py --top-k 5 --score-threshold 0.62

4. 仅运行检索评测（跳过生成评测，更快）:
   uv run python scripts/evaluate_rag.py --retrieval-only

5. 输出详细日志:
   uv run python scripts/evaluate_rag.py --verbose

6. 导出评测结果到文件:
   uv run python scripts/evaluate_rag.py --output results.json

 使用方法

  Linux/WSL：

  SILICONFLOW_API_KEY=你的API密钥 uv run python scripts/evaluate_rag.py

  Windows CMD：

  set SILICONFLOW_API_KEY=你的API密钥
  uv run python scripts/evaluate_rag.py

  Windows PowerShell：

  $env:SILICONFLOW_API_KEY="你的API密钥"
  uv run python scripts/evaluate_rag.py

================================================================================
评测集格式 (JSON):
================================================================================

[
    {
        "query": "LoRA 的核心思想是什么？",
        "ground_truth_answer": "LoRA 通过低秩矩阵分解来高效微调大模型...",
        "relevant_arxiv_ids": ["2106.09685"],
        "expected_keywords": ["low-rank", "adaptation", "fine-tuning"]
    },
    ...
]

字段说明:
- query: 测试问题（必填）
- ground_truth_answer: 标准答案（可选，用于生成质量评估）
- relevant_arxiv_ids: 相关论文的 arXiv ID 列表（必填，用于检索评估）
- expected_keywords: 答案中应包含的关键词（可选，用于关键词覆盖率）

================================================================================
评测指标说明:
================================================================================

检索质量指标:
- Recall@K: 在 TopK 结果中，相关文档的召回率
- Precision@K: TopK 结果中相关文档的比例
- Hit Rate: 至少检索到一个相关文档的查询比例
- MRR (Mean Reciprocal Rank): 第一个正确答案位置倒数的平均值

生成质量指标（需要 LLM）:
- Faithfulness: 答案是否基于检索到的文档（降幻觉）
- Answer Relevancy: 答案与问题的相关程度
- Keyword Coverage: 答案中关键词的覆盖率

引用质量指标:
- Citation Coverage: 检索结果中包含相关论文的比例
- Citation Accuracy: 引用的 arXiv ID 格式正确率

================================================================================
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import requests
from qdrant_client import QdrantClient

# ---- 环境变量 ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

# Ollama Chat 模型（本地回退）
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6b")

# 硅基流动 API 配置（用于生成评测，推荐）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")

# ---- 内置示例评测集 ----
# 可根据实际入库的论文数据调整
BUILTIN_EVAL_DATASET = [
    {
        "query": "What is the core idea of LoRA (Low-Rank Adaptation)?",
        "ground_truth_answer": "LoRA freezes pre-trained model weights and injects trainable low-rank decomposition matrices into each layer, reducing trainable parameters while maintaining model quality.",
        "relevant_arxiv_ids": ["2106.09685"],
        "expected_keywords": ["low-rank", "adaptation", "fine-tuning", "trainable", "frozen"],
    },
    {
        "query": "How does latent diffusion model work?",
        "ground_truth_answer": "Latent diffusion models operate in a compressed latent space learned by an autoencoder, applying the diffusion process there instead of pixel space, which significantly reduces computational requirements.",
        "relevant_arxiv_ids": ["2112.10752"],
        "expected_keywords": ["latent", "diffusion", "autoencoder", "denoising"],
    },
    {
        "query": "What is the transformer architecture?",
        "ground_truth_answer": "The Transformer is based on self-attention mechanisms, dispensing with recurrence and convolutions entirely, consisting of encoder and decoder stacks with multi-head attention.",
        "relevant_arxiv_ids": ["1706.03762"],
        "expected_keywords": ["attention", "self-attention", "encoder", "decoder"],
    },
    {
        "query": "Explain BERT pre-training approach",
        "ground_truth_answer": "BERT uses masked language modeling (MLM) and next sentence prediction (NSP) for pre-training bidirectional representations from unlabeled text.",
        "relevant_arxiv_ids": ["1810.04805"],
        "expected_keywords": ["masked", "bidirectional", "pre-training", "MLM"],
    },
    {
        "query": "What is GPT and how does it generate text?",
        "ground_truth_answer": "GPT is a generative pre-trained transformer that uses autoregressive language modeling, predicting the next token based on previous context.",
        "relevant_arxiv_ids": ["2005.14165", "1801.06146"],
        "expected_keywords": ["autoregressive", "generative", "language model", "pre-trained"],
    },
]


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    avg_score: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0


@dataclass
class GenerationMetrics:
    """生成质量指标"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    keyword_coverage: float = 0.0
    total_evaluated: int = 0


@dataclass
class CitationMetrics:
    """引用质量指标"""
    citation_coverage: float = 0.0
    citation_accuracy: float = 0.0
    total_citations: int = 0


@dataclass
class EvalResult:
    """评测结果汇总"""
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    citation: CitationMetrics = field(default_factory=CitationMetrics)
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


def ollama_embed(text: str) -> List[float]:
    """调用 Ollama 获取文本嵌入向量"""
    r = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    if "embedding" in data:
        return data["embedding"]
    raise ValueError(f"Unexpected response: {list(data.keys())}")


def llm_chat(prompt: str, system: str = "") -> str:
    """
    调用 LLM 生成回复
    优先使用硅基流动 API，如果未配置则回退到 Ollama
    """
    # 如果配置了硅基流动 API Key，使用硅基流动
    if SILICONFLOW_API_KEY:
        return siliconflow_chat(prompt, system)
    else:
        return ollama_chat(prompt, system)


def siliconflow_chat(prompt: str, system: str = "") -> str:
    """
    调用硅基流动 API 生成回复
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    r = requests.post(
        f"{SILICONFLOW_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        },
        json={
            "model": SILICONFLOW_MODEL,
            "messages": messages,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def ollama_chat(prompt: str, system: str = "") -> str:
    """
    调用 Ollama 模型生成回复
    优先尝试 /api/chat，失败则回退到 /api/generate
    """
    base_url = OLLAMA_BASE_URL.rstrip('/')
    
    # 方法1：尝试 /api/chat（新版 Ollama）
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        r = requests.post(
            f"{base_url}/api/chat",
            json={"model": OLLAMA_CHAT_MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "")
    except Exception:
        pass
    
    # 方法2：回退到 /api/generate（兼容旧版）
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    r = requests.post(
        f"{base_url}/api/generate",
        json={"model": OLLAMA_CHAT_MODEL, "prompt": full_prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def get_qdrant_client() -> QdrantClient:
    """获取 Qdrant 客户端"""
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(url=QDRANT_URL, api_key=api_key)
    return QdrantClient(url=QDRANT_URL)


def search_papers(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    检索论文（与 scholar_search_tool 逻辑一致）
    返回去重后的论文列表
    """
    vec = ollama_embed(query.strip())
    qc = get_qdrant_client()
    
    raw_hits = qc.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k * 5,
        with_payload=True,
        score_threshold=score_threshold if score_threshold > 0 else None,
    )
    
    # 按 arxiv_id 去重，保留最高分
    best_by_paper: Dict[str, Any] = {}
    for h in raw_hits:
        p = h.payload or {}
        arxiv_id = p.get("arxiv_id") or p.get("id") or ""
        if not arxiv_id:
            continue
        prev = best_by_paper.get(arxiv_id)
        if prev is None or h.score > prev.score:
            best_by_paper[arxiv_id] = h
    
    results = []
    for h in sorted(best_by_paper.values(), key=lambda x: x.score, reverse=True)[:top_k]:
        p = h.payload or {}
        arxiv_id = p.get("arxiv_id") or p.get("id") or ""
        results.append({
            "arxiv_id": arxiv_id,
            "title": p.get("title", ""),
            "score": float(h.score),
            "snippet": (p.get("chunk") or p.get("summary") or "")[:500],
        })
    
    return results


def evaluate_retrieval(
    eval_data: List[Dict],
    top_k: int = 5,
    score_threshold: float = 0.0,
    verbose: bool = False,
) -> tuple[RetrievalMetrics, List[Dict]]:
    """
    评估检索质量
    
    计算指标:
    - Recall@K: 召回的相关文档数 / 总相关文档数
    - Precision@K: 召回的相关文档数 / K
    - Hit Rate: 至少召回一个相关文档的查询比例
    - MRR: 第一个相关文档位置倒数的平均值
    """
    metrics = RetrievalMetrics(total_queries=len(eval_data))
    detailed = []
    
    total_recall = 0.0
    total_precision = 0.0
    total_hits = 0
    total_rr = 0.0
    total_scores = []
    
    for item in eval_data:
        query = item["query"]
        relevant_ids = set(item.get("relevant_arxiv_ids", []))
        
        if verbose:
            print(f"\n[Query] {query}")
            print(f"  Expected: {relevant_ids}")
        
        try:
            results = search_papers(query, top_k=top_k, score_threshold=score_threshold)
            retrieved_ids = [r["arxiv_id"] for r in results]
            scores = [r["score"] for r in results]
            
            if verbose:
                print(f"  Retrieved: {retrieved_ids}")
                print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
            
            # 计算指标
            relevant_retrieved = relevant_ids & set(retrieved_ids)
            
            # Recall@K
            recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
            total_recall += recall
            
            # Precision@K
            precision = len(relevant_retrieved) / top_k if top_k > 0 else 0
            total_precision += precision
            
            # Hit Rate
            hit = 1 if relevant_retrieved else 0
            total_hits += hit
            
            # MRR
            rr = 0.0
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant_ids:
                    rr = 1.0 / (i + 1)
                    break
            total_rr += rr
            
            # 平均分数
            if scores:
                total_scores.extend(scores)
            
            metrics.successful_queries += 1
            
            detailed.append({
                "query": query,
                "expected_ids": list(relevant_ids),
                "retrieved_ids": retrieved_ids,
                "scores": scores,
                "recall": recall,
                "precision": precision,
                "hit": hit,
                "rr": rr,
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            detailed.append({
                "query": query,
                "error": str(e),
            })
    
    n = metrics.successful_queries
    if n > 0:
        metrics.recall_at_k = total_recall / n
        metrics.precision_at_k = total_precision / n
        metrics.hit_rate = total_hits / n
        metrics.mrr = total_rr / n
    if total_scores:
        metrics.avg_score = sum(total_scores) / len(total_scores)
    
    return metrics, detailed


def evaluate_generation(
    eval_data: List[Dict],
    top_k: int = 5,
    score_threshold: float = 0.0,
    verbose: bool = False,
) -> tuple[GenerationMetrics, List[Dict]]:
    """
    评估生成质量（使用 LLM-as-Judge）
    
    计算指标:
    - Faithfulness: 答案是否基于检索内容
    - Answer Relevancy: 答案与问题的相关性
    - Keyword Coverage: 关键词覆盖率
    """
    metrics = GenerationMetrics()
    detailed = []
    
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_keyword_cov = 0.0
    
    for item in eval_data:
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])
        
        if verbose:
            print(f"\n[Generation Eval] {query}")
        
        try:
            # 1. 检索
            results = search_papers(query, top_k=top_k, score_threshold=score_threshold)
            context = "\n\n".join([
                f"[{r['arxiv_id']}] {r['title']}\n{r['snippet']}"
                for r in results
            ])
            
            # 2. 生成答案
            gen_prompt = f"""Based on the following retrieved paper snippets, answer the question.

Context:
{context}

Question: {query}

Answer (be concise and cite arXiv IDs where relevant):"""
            
            answer = llm_chat(gen_prompt)
            
            if verbose:
                print(f"  Answer: {answer[:200]}...")
            
            # 3. 评估 Faithfulness（LLM-as-Judge）
            faith_prompt = f"""You are evaluating whether an answer is faithful to the given context.

Context:
{context}

Question: {query}
Answer: {answer}

Rate the faithfulness from 0 to 1:
- 1.0: Answer is fully supported by the context
- 0.5: Answer is partially supported
- 0.0: Answer contains information not in context (hallucination)

Respond with ONLY a number between 0 and 1."""
            
            faith_score = 0.5
            try:
                faith_resp = llm_chat(faith_prompt)
                faith_score = float(faith_resp.strip())
                faith_score = max(0, min(1, faith_score))
            except:
                pass
            total_faithfulness += faith_score
            
            # 4. 评估 Answer Relevancy
            rel_prompt = f"""Rate how relevant this answer is to the question.

Question: {query}
Answer: {answer}

Rate from 0 to 1:
- 1.0: Directly answers the question
- 0.5: Partially relevant
- 0.0: Not relevant at all

Respond with ONLY a number between 0 and 1."""
            
            rel_score = 0.5
            try:
                rel_resp = llm_chat(rel_prompt)
                rel_score = float(rel_resp.strip())
                rel_score = max(0, min(1, rel_score))
            except:
                pass
            total_relevancy += rel_score
            
            # 5. 关键词覆盖率
            if expected_keywords:
                answer_lower = answer.lower()
                covered = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
                kw_cov = covered / len(expected_keywords)
            else:
                kw_cov = 1.0
            total_keyword_cov += kw_cov
            
            metrics.total_evaluated += 1
            
            detailed.append({
                "query": query,
                "answer": answer,
                "faithfulness": faith_score,
                "relevancy": rel_score,
                "keyword_coverage": kw_cov,
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            detailed.append({
                "query": query,
                "error": str(e),
            })
    
    n = metrics.total_evaluated
    if n > 0:
        metrics.faithfulness = total_faithfulness / n
        metrics.answer_relevancy = total_relevancy / n
        metrics.keyword_coverage = total_keyword_cov / n
    
    return metrics, detailed


def evaluate_citations(detailed_retrieval: List[Dict]) -> CitationMetrics:
    """评估引用质量"""
    metrics = CitationMetrics()
    
    total_coverage = 0.0
    total_valid = 0
    total_citations = 0
    
    for item in detailed_retrieval:
        if "error" in item:
            continue
        
        expected = set(item.get("expected_ids", []))
        retrieved = set(item.get("retrieved_ids", []))
        
        # Citation Coverage: 检索到的相关论文比例
        if expected:
            coverage = len(expected & retrieved) / len(expected)
            total_coverage += coverage
        
        # Citation Accuracy: arXiv ID 格式验证
        for rid in retrieved:
            total_citations += 1
            # 简单验证 arXiv ID 格式 (例如: 2106.09685)
            if rid and ("." in rid or rid.isdigit()):
                total_valid += 1
    
    n = len([d for d in detailed_retrieval if "error" not in d])
    if n > 0:
        metrics.citation_coverage = total_coverage / n
    if total_citations > 0:
        metrics.citation_accuracy = total_valid / total_citations
    metrics.total_citations = total_citations
    
    return metrics


def print_report(result: EvalResult):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("RAG 评测报告")
    print("=" * 60)
    print(f"时间: {result.timestamp}")
    print(f"配置: top_k={result.config.get('top_k')}, threshold={result.config.get('score_threshold')}")
    
    print("\n【检索质量指标】")
    r = result.retrieval
    print(f"  Recall@K:      {r.recall_at_k:.3f}")
    print(f"  Precision@K:   {r.precision_at_k:.3f}")
    print(f"  Hit Rate:      {r.hit_rate:.3f}")
    print(f"  MRR:           {r.mrr:.3f}")
    print(f"  Avg Score:     {r.avg_score:.3f}")
    print(f"  成功查询:      {r.successful_queries}/{r.total_queries}")
    
    if result.generation.total_evaluated > 0:
        print("\n【生成质量指标】")
        g = result.generation
        print(f"  Faithfulness:     {g.faithfulness:.3f}")
        print(f"  Answer Relevancy: {g.answer_relevancy:.3f}")
        print(f"  Keyword Coverage: {g.keyword_coverage:.3f}")
    
    print("\n【引用质量指标】")
    c = result.citation
    print(f"  Citation Coverage:  {c.citation_coverage:.3f}")
    print(f"  Citation Accuracy:  {c.citation_accuracy:.3f}")
    print(f"  Total Citations:    {c.total_citations}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="评测数据集 JSON 文件路径（默认使用内置示例）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索返回的论文数量（默认: 5）",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="相似度阈值（默认: 0，即不过滤）",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="仅运行检索评测（跳过生成评测）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出结果到 JSON 文件",
    )
    args = parser.parse_args()
    
    # 加载评测数据
    if args.eval_file:
        with open(args.eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"已加载评测集: {args.eval_file} ({len(eval_data)} 条)")
    else:
        eval_data = BUILTIN_EVAL_DATASET
        print(f"使用内置示例评测集 ({len(eval_data)} 条)")
    
    # 初始化结果
    result = EvalResult(
        timestamp=datetime.now().isoformat(),
        config={
            "top_k": args.top_k,
            "score_threshold": args.score_threshold,
            "eval_file": args.eval_file,
            "retrieval_only": args.retrieval_only,
        },
    )
    
    # 检索评测
    print("\n[1/3] 运行检索评测...")
    result.retrieval, retrieval_detailed = evaluate_retrieval(
        eval_data,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        verbose=args.verbose,
    )
    result.detailed_results.extend(retrieval_detailed)
    
    # 引用评测
    print("\n[2/3] 计算引用指标...")
    result.citation = evaluate_citations(retrieval_detailed)
    
    # 生成评测
    if not args.retrieval_only:
        print("\n[3/3] 运行生成评测（LLM-as-Judge）...")
        result.generation, gen_detailed = evaluate_generation(
            eval_data,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            verbose=args.verbose,
        )
    else:
        print("\n[3/3] 跳过生成评测（--retrieval-only）")
    
    # 打印报告
    print_report(result)
    
    # 导出结果
    if args.output:
        output_data = {
            "timestamp": result.timestamp,
            "config": result.config,
            "retrieval": asdict(result.retrieval),
            "generation": asdict(result.generation),
            "citation": asdict(result.citation),
            "detailed_results": result.detailed_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
