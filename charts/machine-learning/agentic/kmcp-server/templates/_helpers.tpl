{{/*
Validate that the MCP server name ends in "-mcp" (naming convention parallel to
the kagent-agent chart's "-agent" suffix; keeps IRSA/discovery naming consistent).
*/}}
{{- define "kmcp-server.validateName" -}}
{{- if not (hasSuffix "-mcp" .Values.name) -}}
{{- fail "MCP server name must end in '-mcp'" -}}
{{- end -}}
{{- end -}}

{{/*
Validate required fields.
*/}}
{{- define "kmcp-server.validateRequired" -}}
{{- if not .Values.name -}}
{{- fail "name is required" -}}
{{- end -}}
{{- if not .Values.image.repository -}}
{{- fail "image.repository is required" -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "kmcp-server.labels" -}}
app.kubernetes.io/name: {{ .Values.name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
