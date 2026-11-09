{{- /*
Deliberately a COPY of the sibling chart's helpers, not an import.

Helm has no mechanism for sharing a partial between two charts without making
one a subchart of the other, and a subchart relationship between two serving
stacks would be a lie about the dependency. So the MPS machinery below is
duplicated verbatim from vllm-openai, and it MUST stay verbatim: it encodes
facts about MPS and the node, which are the same facts for both stacks. If one
copy changes and the other does not, the two charts stop being comparable and
every measured delta between them becomes unreadable.

What legitimately differs from the sibling: the template prefix, and
`vllm-tis.validate`, which reads tensor_parallel_size out of this stack's own
values block. Nothing else.
*/ -}}

{{- define "vllm-tis.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- /*
Release name + tenant name. IDENTICAL in shape to the sibling chart's, and it has
to be: the load driver builds a tenant's URL from
  http://<release>-<name>.<namespace>.svc.cluster.local:<port>
without knowing which chart served it. A different fullname here would give the
driver nothing to talk to, and the failure would look like a dead server.
*/ -}}
{{- define "vllm-tis.fullname" -}}
{{- printf "%s-%s" .Release.Name .Values.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vllm-tis.labels" -}}
app.kubernetes.io/name: {{ include "vllm-tis.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
colocation/tenant: {{ .Values.name }}
{{- end -}}

{{- define "vllm-tis.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vllm-tis.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
colocation/tenant: {{ .Values.name }}
{{- end -}}

{{- /*
The tenant's directory inside the generated model repository. One model per pod,
and its directory name is the served model name, because Triton's OpenAI
frontend advertises a model under its TRITON model name. So there is exactly one
string to get right and it is `model.served_name`.
*/ -}}
{{- define "vllm-tis.modelDir" -}}
{{- printf "%s/%s" .Values.triton.modelRepositoryPath .Values.model.served_name -}}
{{- end -}}

{{- /*
CUDA_VISIBLE_DEVICES: the tenant's host GPU ordinals. Isolation happens HERE,
client-side, while NVIDIA_VISIBLE_DEVICES stays "all" (design §1.9). The device
plugin is bypassed entirely, so nothing else scopes the tenant to its GPUs.

On this stack there is a second party that can overwrite this variable, and it
is why config.pbtxt says KIND_MODEL. With KIND_GPU the vLLM backend does
`os.environ["CUDA_VISIBLE_DEVICES"] = str(triton_device_id)` — Triton's OWN
device id, which has nothing to do with the placement — and the tenant would run
on whichever GPU Triton picked, with the envelope pinned to a UUID it never
touches. KIND_MODEL without a GPU_DEVICE_IDS parameter leaves the variable
alone, which is what the isolation depends on. (Read out of
/opt/tritonserver/backends/vllm/model.py::_validate_device_config.)
*/ -}}
{{- define "vllm-tis.cudaVisibleDevices" -}}
{{- if not .Values.placement.gpu_indices -}}
{{- fail "placement.gpu_indices must be a non-empty list" -}}
{{- end -}}
{{- join "," .Values.placement.gpu_indices -}}
{{- end -}}

{{- /*
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT, keyed by GPU UUID.

  GPU-00040000-...=8G,GPU-00070000-...=8G

NEVER ordinals. Early hardware testing recorded the failure mode: with the key
"1=6G" the limit was silently ignored and mem_get_info() read back the full
physical capacity while the MPS server logged contexts on its own private
indices. There is no error, no warning, and no way to detect it except by
measuring — which is what the verification init container exists to do.

The list length must equal tensor_parallel_size: every shard gets the same
envelope, since the solver's per-tenant envelope is per-GPU.
*/ -}}
{{- define "vllm-tis.mpsMemLimit" -}}
{{- $uuids := .Values.placement.gpu_uuids -}}
{{- if not $uuids -}}
{{- fail "placement.gpu_uuids must be a non-empty list of GPU UUIDs (never ordinals — an ordinal key fails silently)" -}}
{{- end -}}
{{- $limit := .Values.mps.memory_limit_gib -}}
{{- if not $limit -}}
{{- fail "mps.memory_limit_gib must be set" -}}
{{- end -}}
{{- $parts := list -}}
{{- range $uuids -}}
{{- if not (hasPrefix "GPU-" .) -}}
{{- fail (printf "placement.gpu_uuids entry %q is not a GPU UUID. It must look like GPU-00010000-0000-0000-0000-000000000001 — an ordinal key fails SILENTLY" .) -}}
{{- end -}}
{{- $parts = append $parts (printf "%s=%dG" . (int $limit)) -}}
{{- end -}}
{{- join "," $parts -}}
{{- end -}}

{{- /*
The MPS environment, shared by the verification init container and the server
container. Defined once because the two MUST agree: if the init container
verifies under a different environment than the server runs under, it verifies
nothing (impl §4.4).

Note NVIDIA_VISIBLE_DEVICES=all with isolation via CUDA_VISIBLE_DEVICES. This is
the reverse of the obvious arrangement and is required (design §1.9): it also
means the value reads back as "void" in-container, so a tenant cannot
self-discover its assignment.
*/ -}}
{{- define "vllm-tis.mpsEnv" -}}
- name: NVIDIA_VISIBLE_DEVICES
  value: "all"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "compute,utility"
- name: CUDA_VISIBLE_DEVICES
  value: {{ include "vllm-tis.cudaVisibleDevices" . | quote }}
- name: CUDA_MPS_PIPE_DIRECTORY
  value: {{ .Values.mps.pipeMountPath | quote }}
- name: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
  value: {{ include "vllm-tis.mpsMemLimit" . | quote }}
{{- if .Values.mps.active_thread_percentage }}
{{- /* An SM ceiling, not a reservation, and UNTESTED (test spec §14). Left
       unset by design; set it only deliberately. */}}
- name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: {{ .Values.mps.active_thread_percentage | quote }}
{{- end }}
{{- end -}}

{{- /*
Free memory the verification init container must observe, in GiB. Defaults to
the nominal envelope; see values.yaml for why the tolerance is 0.5.
*/ -}}
{{- define "vllm-tis.expectedFreeGib" -}}
{{- if .Values.verification.expected_free_gib -}}
{{- .Values.verification.expected_free_gib -}}
{{- else -}}
{{- .Values.mps.memory_limit_gib -}}
{{- end -}}
{{- end -}}

{{- /*
Consistency gate. These are cheap to check at template time and expensive to
diagnose at runtime, so they fail the render instead.
*/ -}}
{{- define "vllm-tis.validate" -}}
{{- $args := .Values.triton_vllm.engine_args -}}
{{- if not $args -}}
{{- fail "triton_vllm.engine_args must be set: it is the vLLM AsyncEngineArgs this tenant's model.json is written from" -}}
{{- end -}}
{{- if not $args.tensor_parallel_size -}}
{{- fail "triton_vllm.engine_args.tensor_parallel_size must be set" -}}
{{- end -}}
{{- $tp := int $args.tensor_parallel_size -}}
{{- if ne (len .Values.placement.gpu_indices) $tp -}}
{{- fail (printf "placement.gpu_indices has %d entries but triton_vllm.engine_args.tensor_parallel_size is %d — a TP group's shards must occupy exactly that many distinct GPUs" (len .Values.placement.gpu_indices) $tp) -}}
{{- end -}}
{{- if ne (len .Values.placement.gpu_uuids) $tp -}}
{{- fail (printf "placement.gpu_uuids has %d entries but triton_vllm.engine_args.tensor_parallel_size is %d — one UUID per shard, in the same order as gpu_indices" (len .Values.placement.gpu_uuids) $tp) -}}
{{- end -}}
{{- if not .Values.placement.node -}}
{{- fail "placement.node must be set: the GPU-index->UUID mapping is node-specific, so a configuration is not portable across nodes (impl §7)" -}}
{{- end -}}
{{- /* The weights path has exactly one home, `model.path`, which is what the
       chart writes into model.json and passes as --tokenizer. Restating it under
       engine_args would let the two disagree, and the engine would then serve
       one model's weights with another's tokenizer — which does not fail, it
       just produces garbage. */ -}}
{{- if hasKey $args "model" -}}
{{- fail "triton_vllm.engine_args must not set `model`: the chart writes it from model.path, so that the weights the engine loads and the tokenizer the frontend loads cannot drift apart" -}}
{{- end -}}
{{- if hasKey $args "served_model_name" -}}
{{- fail "triton_vllm.engine_args must not set `served_model_name`: this frontend serves a model under its Triton model name, which this chart takes from model.served_name" -}}
{{- end -}}
{{- if not .Values.model.served_name -}}
{{- fail "model.served_name must be set: it is both the model id clients send and the directory name inside the generated Triton model repository" -}}
{{- end -}}
{{- end -}}
