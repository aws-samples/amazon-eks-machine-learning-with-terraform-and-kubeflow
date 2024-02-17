from kfp import dsl
from kfp import compiler
from kfp import components

from typing import List, Dict

helm_charts_component = components.load_component_from_file('kfp/components/packages/helm_charts_component.yaml')

@dsl.pipeline
def helm_charts_pipeline(chart_configs: List[Dict]) -> str:
    helm_charts_task = helm_charts_component(chart_configs=chart_configs)
    return helm_charts_task.output

compiler.Compiler().compile(helm_charts_pipeline, package_path='kfp/pipelines/packages/helm_charts_pipeline.yaml')