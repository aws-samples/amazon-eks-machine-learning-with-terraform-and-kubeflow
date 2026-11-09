{{- define "mps-daemon.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mps-daemon.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "mps-daemon.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mps-daemon.labels" -}}
app.kubernetes.io/name: {{ include "mps-daemon.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "mps-daemon.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mps-daemon.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- /*
Base name for the per-node UUID ConfigMap. The node name is appended at
runtime inside the container (it is not known at template time), so RBAC below
grants create namespace-wide and get/update/patch on the "<base>-*" set via a
namespace-wide fallback. See rbac.yaml.
*/ -}}
{{- define "mps-daemon.uuidConfigMapBase" -}}
{{- if .Values.uuidConfigMap.name -}}
{{- .Values.uuidConfigMap.name -}}
{{- else -}}
{{- printf "%s-gpu-uuids" .Release.Name | trunc 55 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "mps-daemon.serviceAccountName" -}}
{{- printf "%s-sa" (include "mps-daemon.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- /*
--- dcgm-exporter ------------------------------------------------------------

Named off .Release.Name rather than off mps-daemon.fullname, matching the
uuidConfigMap convention above, for one reason that is not cosmetic: the harness
has to construct this Service's DNS name in Python before the release exists
(src/colocation/agent/render.py, `exporter_service_url`). A name that is a
one-line function of the release name is a name the agent cannot get wrong.

EVERY exporter resource is release-prefixed, and that is the whole reason these
templates exist instead of a dependency on the upstream dcgm-exporter chart.
That chart names its metrics ConfigMap `exporter-metrics-config-map` and its Role
`dcgm-exporter-read-cm` -- fixed strings, no release prefix. A second per-run
release in the same namespace therefore fails install with "resource already
exists and is not owned by this release", which makes parallel runs impossible.
The pod spec below is otherwise a transcription of what that chart renders from
observability/dcgm-exporter-values.yaml, i.e. of the configuration already proven
to publish all the DCP fields on L40S under MPS -- see that file for why each
setting is what it is. Two deliberate departures are marked DEPARTURE below.
*/ -}}
{{- define "mps-daemon.exporterName" -}}
{{- printf "%s-dcgm-exporter" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mps-daemon.exporterMetricsConfigMap" -}}
{{- printf "%s-dcgm-metrics" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- /*
NOT app.kubernetes.io/name: mps-daemon. If the exporter pod carried the daemon's
name+instance pair it would match the daemon Deployment's own selector, and that
Deployment would adopt it and then scale it away as a surplus replica.
*/ -}}
{{- define "mps-daemon.exporterSelectorLabels" -}}
app.kubernetes.io/name: dcgm-exporter
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: dcgm-exporter
{{- end -}}

{{- define "mps-daemon.exporterLabels" -}}
{{ include "mps-daemon.exporterSelectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: {{ include "mps-daemon.name" . }}
{{- end -}}
