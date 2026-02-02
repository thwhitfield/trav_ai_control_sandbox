python scripts/ai_control_paper_copied.py \
    --limit 200 \
    --audit-budget 1.0 \
    --defer-thresholds "0.0,0.2,0.5,0.8,1.0" \
    --trusted-model "openai/gpt-4.1-mini-2025-04-14" \
    --untrusted-model "anthropic/claude-haiku-4-5-20251001" \
    --skip-untrusted-monitor-defer \
    --skip-trusted-monitor-edit