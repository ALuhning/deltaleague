"""Agentic workspace episodes: pinned tools, JSON-action loop, budgets and grading."""
import json
from pathlib import Path

import pytest

from delta_cli.public_eval import agentic
from delta_cli.public_eval.runner import serve_args

GENERATION = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5, 'enable_thinking': True}


def task(**extra):
    return {'id': 'a' * 24, 'prompt': 'Find the total.', 'answer': {'total': 7},
            'budget': {'max_turns': 6, 'max_tool_calls': 4, 'max_tokens_per_turn': 100, 'max_generated_tokens': 1000},
            'workspace': {'documents': {
                'entry': {'kind': 'document', 'title': 'Index', 'content': {'table': 't'}},
                't': {'kind': 'table', 'title': 'Rows', 'rows': [{'k': 'x', 'v': 3}, {'k': 'y', 'v': 4}]}}},
            **extra}


class Reply:
    def __init__(self, text=None, status=200, tokens=10, body=None):
        self.status_code = status
        self.body = body or {'choices': [{'message': {'content': f'<think>plan</think>{text}'}, 'finish_reason': 'stop'}],
                             'usage': {'completion_tokens': tokens, 'prompt_tokens': 50}}

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def scripted(*replies):
    sent = []
    def post(url, body):
        sent.append(json.loads(json.dumps(body)))
        return replies[len(sent) - 1]
    return post, sent


def test_runtime_is_the_calibrated_one():
    assert agentic.runtime_sha256() == agentic.RUNTIME_SHA256


def test_episode_uses_real_tools_and_grades_the_final_answer():
    post, sent = scripted(
        Reply(json.dumps({'tool': 'read', 'arguments': {'ref': 'entry'}})),
        Reply(json.dumps({'calls': [{'tool': 'query', 'arguments': {'ref': 't'}},
                                    {'tool': 'calculate', 'arguments': {'expression': '3 + 4'}}]})),
        Reply('```json\n{"final": {"total": 7}}\n```'))
    row, evidence = agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=5, post=post)
    assert row['passed'] and row['outcome'] == 'answered' and row['tool_calls'] == 3 and row['turns'] == 3
    assert evidence['tool_trace'][2]['result'] == {'result': 7}
    # Thinking is stripped from history; sampling, seed and the per-turn budget are sent.
    assert sent[1]['messages'][2] == {'role': 'assistant', 'content': json.dumps({'tool': 'read', 'arguments': {'ref': 'entry'}})}
    assert sent[0]['seed'] == 5 and sent[0]['top_k'] == 20 and sent[0]['max_tokens'] == 100
    assert sent[0]['messages'][0]['content'] == agentic.SYSTEM


def test_wrong_answers_budgets_and_context_are_model_outcomes():
    post, _ = scripted(Reply('{"final": {"total": 8}}'))
    assert agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=1, post=post)[0]['reason'] == 'wrong_answer'
    # Booleans never equal integers.
    post, _ = scripted(Reply('{"final": {"total": true}}'))
    assert not agentic.run_episode(task(answer={'total': 1}), endpoint='u', generation=GENERATION, seed=1, post=post)[0]['passed']
    post, sent = scripted(*[Reply('not json', tokens=400)] * 3)
    row, _ = agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=1, post=post)
    assert row['outcome'] == 'generation_limit' and len(sent) == 3 and sent[2]['max_tokens'] == 100
    assert 'protocol_error' in sent[1]['messages'][-1]['content']
    calls = {'calls': [{'tool': 'read', 'arguments': {'ref': 'entry'}}] * 5}
    post, _ = scripted(Reply(json.dumps(calls)))
    assert agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=1, post=post)[0]['reason'] == 'tool_limit'
    post, _ = scripted(Reply(status=400, body={'error': {'message': "This model's maximum context length is 65536 tokens"}}))
    assert agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=1, post=post)[0]['reason'] == 'context_limit'


def test_server_failures_are_never_scored():
    post, _ = scripted(Reply(status=503, body={'error': 'down'}))
    with pytest.raises(RuntimeError):
        agentic.run_episode(task(), endpoint='u', generation=GENERATION, seed=1, post=post)
    with pytest.raises(ValueError):
        agentic.run_episode(task(budget={'max_turns': 1}), endpoint='u', generation=GENERATION, seed=1, post=post)


def test_tools_see_only_the_public_workspace():
    from delta_cli.public_eval.agentic_runtime import Workspace
    ws = Workspace(task()['workspace'])
    assert 'error' in ws.call('read', {'ref': 'answer'})
    assert 'error' in ws.call('calculate', {'expression': "__import__('os')"})


def test_serve_args_match_the_pinned_runtime_and_engine_profiles():
    runtime = json.loads((Path(agentic.__file__).parent / 'data/runtime.json').read_text())
    assert sorted(serve_args()) == sorted(runtime['server_args'])
    args = serve_args({'max_model_len': 65536, 'engine': {'gpu_memory_utilization': .93, 'kv_cache_dtype': 'fp8',
                                                          'max_num_seqs': 64, 'likelihood_gpu_memory_utilization': .93}})
    assert args[args.index('--max-model-len') + 1] == '65536' and args[args.index('--kv-cache-dtype') + 1] == 'fp8'
    assert args[args.index('--gpu-memory-utilization') + 1] == '0.93' and args[args.index('--max-num-seqs') + 1] == '64'


def test_serve_command_publishes_on_localhost_only(tmp_path):
    from typer.testing import CliRunner
    from delta_cli.main import app
    config = tmp_path / 'evaluation.json'
    config.write_text(json.dumps({'generation': {'engine': {'max_num_seqs': 64}}}))
    result = CliRunner().invoke(app, ['public-eval', 'serve-command', '--evaluation-config', str(config),
                                      '--model-dir', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert '-p 127.0.0.1:8000:8000' in result.output and '--max-num-seqs 64' in result.output


def test_private_suite_track_reports_a_trusted_pass(monkeypatch):
    import httpx
    from delta_cli.public_eval.custom_suite import evaluate
    replies = iter([Reply('{"final": {"total": 7}}'), Reply('{"final": {"total": 1}}')])
    monkeypatch.setattr(httpx, 'post', lambda url, json=None, timeout=None: next(replies))
    tasks = [{**task(), 'track': 'agentic'}, {**task(id='b' * 24), 'track': 'agentic'}]
    rows, evidence = evaluate(tasks, concurrency=1, endpoint='u', generation=GENERATION, seed=1, root=None,
                              tokenizer=None, harbor_root=None, image=None)
    assert [r['passed'] for r in rows] == [True, False] and all(r['state'] == 'complete' for r in rows)
    assert rows[0]['track'] == 'agentic' and evidence['a' * 24]['final'] == {'total': 7}
    with pytest.raises(ValueError):
        unanswered = {**{k: v for k, v in task().items() if k != 'answer'}, 'track': 'agentic'}
        evaluate([unanswered],
                 endpoint='u', generation=GENERATION, seed=1, root=None, tokenizer=None, harbor_root=None, image=None)
