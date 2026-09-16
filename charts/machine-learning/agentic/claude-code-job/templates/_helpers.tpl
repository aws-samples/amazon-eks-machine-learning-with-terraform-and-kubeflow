{{/*
Validate that the Job name ends in "-job" (naming convention parallel to the
kagent-agent chart's "-agent" and kmcp-server's "-mcp" suffixes).
*/}}
{{- define "claude-code-job.validateName" -}}
{{- if not (hasSuffix "-job" .Values.name) -}}
{{- fail "Job name must end in '-job'" -}}
{{- end -}}
{{- end -}}

{{/*
Validate required fields.
*/}}
{{- define "claude-code-job.validateRequired" -}}
{{- if not .Values.name -}}
{{- fail "name is required" -}}
{{- end -}}
{{- if not .Values.image.repository -}}
{{- fail "image.repository is required" -}}
{{- end -}}
{{- if not .Values.task -}}
{{- fail "task is required (the command/prompt for `claude -p`)" -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "claude-code-job.labels" -}}
app.kubernetes.io/name: {{ .Values.name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
The ServiceAccount name to use on the pod: an explicitly provided name, otherwise the
chart-created one (named after the Job) when annotations are supplied, otherwise "default".
*/}}
{{- define "claude-code-job.serviceAccountName" -}}
{{- if .Values.serviceAccountName -}}
{{- .Values.serviceAccountName -}}
{{- else if .Values.serviceAccountAnnotations -}}
{{- .Values.name -}}
{{- else -}}
default
{{- end -}}
{{- end -}}
