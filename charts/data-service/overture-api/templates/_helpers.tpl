{{/*
=============================================================================
overture-api chart helpers
=============================================================================
*/}}

{{/*
Standard labels applied to all resources.
*/}}
{{- define "overture-api.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: overture-api
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: data-service
{{- end }}

{{/*
Selector labels — used in Deployment.spec.selector and Service.spec.selector.
*/}}
{{- define "overture-api.selectorLabels" -}}
app.kubernetes.io/name: overture-api
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Local FSx path to the bbox-filtered parquet file the serve container reads.
The init job's DuckDB job reads Overture data directly from S3 via httpfs
and writes the bbox-filtered result to this path.
*/}}
{{- define "overture-api.filteredPath" -}}
{{ .Values.fsx.mountPath }}/{{ .Values.fsx.dbDir }}/{{ .Release.Name }}/filtered/segments.parquet
{{- end }}

{{/*
The bounding box rendered as comma-separated floats for the wrapper service env.
*/}}
{{- define "overture-api.bboxString" -}}
{{ index .Values.overture.bbox 0 }},{{ index .Values.overture.bbox 1 }},{{ index .Values.overture.bbox 2 }},{{ index .Values.overture.bbox 3 }}
{{- end }}

{{/*
Cluster-internal endpoint URL for the wrapper service.
*/}}
{{- define "overture-api.endpointURL" -}}
http://{{ .Release.Name }}.{{ .Values.namespace }}.svc.cluster.local
{{- end }}

{{/*
Status endpoint — used by readiness/liveness probes and CFN gate.
*/}}
{{- define "overture-api.statusURL" -}}
{{ include "overture-api.endpointURL" . }}/api/status
{{- end }}
