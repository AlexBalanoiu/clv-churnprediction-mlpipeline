{{- define "clv-api.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "clv-api.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "clv-api.labels" -}}
app.kubernetes.io/name: {{ include "clv-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "clv-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "clv-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
