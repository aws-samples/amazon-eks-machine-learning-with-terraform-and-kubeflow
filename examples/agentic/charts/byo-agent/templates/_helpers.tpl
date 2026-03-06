{{/*
Validate that the agent name ends in "-agent" (required by IRSA trust policy).
*/}}
{{- define "byo-agent.validateName" -}}
{{- if not (hasSuffix "-agent" .Values.name) -}}
{{- fail "agent name must end in '-agent' (required by IRSA trust policy)" -}}
{{- end -}}
{{- end -}}

{{/*
Validate required fields.
*/}}
{{- define "byo-agent.validateRequired" -}}
{{- if not .Values.name -}}
{{- fail "name is required" -}}
{{- end -}}
{{- if not .Values.description -}}
{{- fail "description is required" -}}
{{- end -}}
{{- if not .Values.image.repository -}}
{{- fail "image.repository is required" -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "byo-agent.labels" -}}
app.kubernetes.io/name: {{ .Values.name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
