{{- /*
Deliberately a COPY of the sibling charts' helpers, not an import, for the
reason vllm-triton-inference-server/templates/_helpers.tpl gives: Helm cannot
share a partial between two charts without a subchart relationship, and one
serving stack is not a dependency of another. The MPS machinery below MUST stay
verbatim with the siblings'. It encodes facts about MPS and the node.

What legitimately differs: the template prefix, `dynamo-vllm.validate`, which
reads this stack's own values block, and the helpers that exist because a
tenant here is two pods rather than one (`podLabels`).
*/ -}}

{{- define "dynamo-vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- /*
Release name + tenant name, the same shape as the siblings'. The load driver
builds a tenant's URL as http://<release>-<name>.<namespace>.svc.cluster.local:<port>
without knowing which chart served it, and here that Service fronts the Dynamo
FRONTEND pod. It is also the DynamoGraphDeployment's name, so every object the
operator makes for this tenant is prefixed with it.
*/ -}}
{{- define "dynamo-vllm.fullname" -}}
{{- printf "%s-%s" .Release.Name .Values.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "dynamo-vllm.labels" -}}
app.kubernetes.io/name: {{ include "dynamo-vllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
colocation/tenant: {{ .Values.name }}
{{- end -}}

{{- /*
The labels every pod of this tenant carries, BOTH pods. They are what the
agent's readiness gate and its teardown wait select on
(`agent/deploy.py`: `-l colocation/tenant`, then `app.kubernetes.io/instance`
== release), and here the pods are made by the Dynamo operator rather than by
a Deployment this chart owns -- so the labels have to be written into each
component's podTemplate, which the operator copies onto the pod. A pod missing
them is a pod the gate cannot see: it would pass readiness without the worker,
and teardown would return while the worker still held its envelope.
*/ -}}
{{- define "dynamo-vllm.podLabels" -}}
app.kubernetes.io/name: {{ include "dynamo-vllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
colocation/tenant: {{ .Values.name }}
{{- end -}}

{{- /*
CUDA_VISIBLE_DEVICES: the tenant's host GPU ordinals. Isolation happens HERE,
client-side, while NVIDIA_VISIBLE_DEVICES stays "all" (design §1.9). The device
plugin is bypassed entirely, so nothing else scopes the tenant to its GPUs.
*/ -}}
{{- define "dynamo-vllm.cudaVisibleDevices" -}}
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
{{- define "dynamo-vllm.mpsMemLimit" -}}
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
{{- define "dynamo-vllm.mpsEnv" -}}
- name: NVIDIA_VISIBLE_DEVICES
  value: "all"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "compute,utility"
- name: CUDA_VISIBLE_DEVICES
  value: {{ include "dynamo-vllm.cudaVisibleDevices" . | quote }}
- name: CUDA_MPS_PIPE_DIRECTORY
  value: {{ .Values.mps.pipeMountPath | quote }}
- name: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
  value: {{ include "dynamo-vllm.mpsMemLimit" . | quote }}
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
{{- define "dynamo-vllm.expectedFreeGib" -}}
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
{{- define "dynamo-vllm.validate" -}}
{{- $tp := int .Values.dynamo_vllm.tensor_parallel_size -}}
{{- if ne (len .Values.placement.gpu_indices) $tp -}}
{{- fail (printf "placement.gpu_indices has %d entries but dynamo_vllm.tensor_parallel_size is %d — a TP group's shards must occupy exactly that many distinct GPUs" (len .Values.placement.gpu_indices) $tp) -}}
{{- end -}}
{{- if ne (len .Values.placement.gpu_uuids) $tp -}}
{{- fail (printf "placement.gpu_uuids has %d entries but dynamo_vllm.tensor_parallel_size is %d — one UUID per shard, in the same order as gpu_indices" (len .Values.placement.gpu_uuids) $tp) -}}
{{- end -}}
{{- if not .Values.placement.node -}}
{{- fail "placement.node must be set: the GPU-index->UUID mapping is node-specific, so a configuration is not portable across nodes (impl §7)" -}}
{{- end -}}
{{- if not .Values.model.served_name -}}
{{- fail "model.served_name must be set: it is the model id the frontend routes by, and the frontend's readiness probe waits for exactly that name" -}}
{{- end -}}
{{- /* The adapter refuses these too (engines/dynamo_vllm.py). Repeated here for
       a hand install, which never passes through the adapter. */ -}}
{{- range .Values.dynamo_vllm.extra_args -}}
{{- $flag := regexReplaceAll "[= ].*$" (toString .) "" -}}
{{- if has $flag (list "--disaggregation-mode" "--is-prefill-worker" "--is-decode-worker") -}}
{{- fail (printf "dynamo_vllm.extra_args carries %q: a tenant of this chart is ONE aggregated worker, and a disaggregated half needs a second envelope the solver never priced" .) -}}
{{- end -}}
{{- if has $flag (list "--model" "--served-model-name") -}}
{{- fail (printf "dynamo_vllm.extra_args carries %q, which the chart sets from model.path / model.served_name" .) -}}
{{- end -}}
{{- end -}}
{{- end -}}
