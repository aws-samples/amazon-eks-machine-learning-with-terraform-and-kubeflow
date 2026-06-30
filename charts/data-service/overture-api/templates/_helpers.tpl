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
S3 source path for the Overture parquet partition.
Resolves to: <prefix>/<release>/theme=<theme>/type=<type>/
*/}}
{{- define "overture-api.s3SourcePath" -}}
{{ .Values.overture.s3SourcePrefix }}/{{ .Values.overture.release }}/theme={{ .Values.overture.theme }}/type={{ .Values.overture.type }}
{{- end }}

{{/*
Local FSx path where the raw downloaded parquet shards are staged.
*/}}
{{- define "overture-api.rawDir" -}}
{{ .Values.fsx.mountPath }}/{{ .Values.fsx.dbDir }}/{{ .Release.Name }}/raw
{{- end }}

{{/*
Local FSx path to the bbox-filtered parquet file the serve container reads.
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
