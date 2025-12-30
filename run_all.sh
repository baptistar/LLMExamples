#!/bin/bash

echo "====== Running COT ========"
python qwen_COT.py
echo "====== Running Q&A ========"
python qwen_QandA.py
echo "====== Running Translation ========"
python qwen_translation.py
echo "====== Running Summarizing ========"
python qwen_summarization.py
echo "====== Running Coding ========"
python qwen_coding.py
echo "====== Running VQA ========"
python qwen_vqa.py
echo "====== Running Math ========"
python qwen_math.py
echo "====== Running Math (Stochastic) ========"
python qwen_math_stochastic.py
