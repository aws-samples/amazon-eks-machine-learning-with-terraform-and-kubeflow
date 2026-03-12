{{/*
=============================================================================
overpass-api chart helpers
=============================================================================
*/}}

{{/*
Standard labels applied to all resources.
*/}}
{{- define "overpass-api.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: overpass-api
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: data-service
{{- end }}

{{/*
Selector labels — used in Deployment.spec.selector and Service.spec.selector.
Intentionally minimal: only name + instance to avoid selector immutability issues.
*/}}
{{- define "overpass-api.selectorLabels" -}}
app.kubernetes.io/name: overpass-api
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Derive the local PBF filename from the URL.
If pbf.filenameOverride is set, use that.
Otherwise extract the last path segment from the URL.

Usage: include "overpass-api.pbfFilename" .
*/}}
{{- define "overpass-api.pbfFilename" -}}
{{- if .Values.pbf.filenameOverride -}}
  {{- .Values.pbf.filenameOverride -}}
{{- else -}}
  {{- .Values.pbf.url | base -}}
{{- end -}}
{{- end }}

{{/*
Full FSx path to the PBF file.
*/}}
{{- define "overpass-api.pbfPath" -}}
{{ .Values.fsx.mountPath }}/{{ .Values.fsx.dbDir }}/{{ .Release.Name }}/pbf/{{ include "overpass-api.pbfFilename" . }}
{{- end }}

{{/*
Full FSx path to the Overpass database directory for this release.
Each release gets its own subdirectory to avoid collision when multiple
releases share the same FSx volume.
*/}}
{{- define "overpass-api.dbPath" -}}
{{ .Values.fsx.mountPath }}/{{ .Values.fsx.dbDir }}/{{ .Release.Name }}
{{- end }}

{{/*
Overpass API endpoint URL for this release.
Used in the ConfigMap so agents can discover the endpoint by name.
*/}}
{{- define "overpass-api.endpointURL" -}}
http://{{ .Release.Name }}.{{ .Values.namespace }}.svc.cluster.local/api/interpreter
{{- end }}

{{/*
Overpass API status URL — used in readiness/liveness probes.
*/}}
{{- define "overpass-api.statusURL" -}}
http://{{ .Release.Name }}.{{ .Values.namespace }}.svc.cluster.local/api/status
{{- end }}
