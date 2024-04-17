{{/*
Expand the name of the chart.
*/}}
{{- define "ser-application.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ser-application.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ser-application.labels" -}}
helm.sh/chart: {{ include "ser-application.chart" . }}
{{ include "ser-application.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ser-application.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ser-application.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service Account Name
*/}}
{{- define "ser-application.serviceAccountName" -}}
{{ default (include "ser-application.name" .) .Values.serviceAccount.name }}
{{- end -}}

{{/*
Fullname Override
*/}}
{{- define "ser-application.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Annotations
*/}}
{{- define "ser-application.annotations" -}}
{{- with .Values.annotations }}
{{- toYaml . | nindent 4 }}
{{- end }}
{{- end }}

{{/* 
Define Airflow common environment variables 
*/}}
{{- define "airflow.commonEnvVars" -}}
- name: AIRFLOW__CORE__EXECUTOR
  value: "CeleryExecutor"
- name: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
  value: "postgresql+psycopg2://airflow:airflow@postgres/airflow"
- name: AIRFLOW__CELERY__RESULT_BACKEND
  value: "db+postgresql://airflow:airflow@postgres/airflow"
- name: AIRFLOW__CELERY__BROKER_URL
  value: "redis://:@redis:6379/0"
- name: AIRFLOW__CORE__FERNET_KEY
  value: ""
- name: AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION
  value: "true"
- name: AIRFLOW__CORE__LOAD_EXAMPLES
  value: "false"
- name: AIRFLOW__API__AUTH_BACKENDS
  value: "airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session"
- name: AIRFLOW__WEBSERVER__SECRET_KEY
  value: "abcdefghij"
- name: AIRFLOW__CORE__EXECUTE_TASKS_NEW_PYTHON_INTERPRETER
  value: "true"
- name: AIRFLOW__CORE__ENABLE_XCOM_PICKLING
  value: "true"
{{- end }}

{{/* Define common volume mounts */}}
{{- define "airflow.commonVolumeMounts" -}}
- name: dags
  mountPath: /opt/airflow/dags
- name: logs
  mountPath: /opt/airflow/logs
- name: config
  mountPath: /opt/airflow/config
- name: plugins
  mountPath: /opt/airflow/plugins
{{- end }}
