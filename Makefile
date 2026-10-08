.PHONY: install start stop health client benchmark benchmark-context benchmark-saturation benchmark-conversation kv-capacity validate validate-context validate-saturation validate-conversation validate-vision-encoder validate-scheduler validate-maxseq validate-admission inference-report analysis ab-cuda-graph ab-prefix-cache ab-vision-encoder ab-scheduler ab-maxseq ab-admission monitoring-up monitoring-down profile-check profile-permissions profile-nsys profile-ncu profile-export profile-compare profile-validate

install:
	./scripts/install_vllm.sh

start:
	./scripts/start_vllm_background.sh

stop:
	./scripts/stop_vllm.sh

health:
	./scripts/healthcheck.sh

client:
	.venv-vllm/bin/python client/multimodal_client.py

benchmark:
	.venv-vllm/bin/python benchmark/run_suite.py

benchmark-context:
	.venv-vllm/bin/python benchmark/context_ttft.py

benchmark-saturation:
	.venv-vllm/bin/python benchmark/saturation.py

benchmark-conversation:
	.venv-vllm/bin/python benchmark/conversation_cache.py

kv-capacity:
	.venv-vllm/bin/python benchmark/kv_capacity.py

validate:
	.venv-vllm/bin/python benchmark/validate_results.py results/vllm_bf16/summary.csv --raw results/vllm_bf16/requests.jsonl

validate-context:
	.venv-vllm/bin/python benchmark/validate_context_ttft.py results/context_ttft

validate-saturation:
	.venv-vllm/bin/python benchmark/validate_saturation.py

validate-conversation:
	.venv-vllm/bin/python benchmark/validate_conversation_cache.py

validate-vision-encoder:
	.venv-vllm/bin/python benchmark/validate_vision_encoder_ab.py

inference-report:
	.venv-vllm/bin/python benchmark/build_inference_report.py

analysis:
	.venv-vllm/bin/python benchmark/build_analysis.py
	.venv-vllm/bin/python -m unittest discover -s tests -v

ab-cuda-graph:
	./scripts/run_cuda_graph_ab.sh

ab-prefix-cache:
	./scripts/run_prefix_cache_ab.sh

ab-vision-encoder:
	./scripts/run_vision_encoder_ab.sh

ab-scheduler:
	./scripts/run_scheduler_ab.sh

ab-maxseq:
	./scripts/run_maxseq_ab.sh

ab-admission:
	./scripts/run_admission_ab.sh

validate-scheduler:
	.venv-vllm/bin/python benchmark/validate_scheduler.py

validate-maxseq:
	.venv-vllm/bin/python benchmark/validate_maxseq.py

validate-admission:
	.venv-vllm/bin/python benchmark/validate_admission.py

profile-check:
	./scripts/profile_env.sh

profile-permissions:
	./scripts/profile_permissions.sh status

profile-nsys:
	./scripts/profile_nsys_offline.sh

profile-ncu:
	./scripts/profile_ncu.sh

profile-export:
	./scripts/profile_export.sh

profile-compare:
	.venv-vllm/bin/python profiling/compare.py

profile-validate:
	.venv-vllm/bin/python profiling/validate_profiles.py $(if $(filter 1,$(PROFILE_REQUIRE_NCU)),--require-ncu,)

monitoring-up:
	docker compose -f monitoring/docker-compose.yml up -d

monitoring-down:
	docker compose -f monitoring/docker-compose.yml down
