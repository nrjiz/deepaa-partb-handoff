"""Route B final artifacts: scoped provenance plus unchanged numerical validation."""
from .workflow_evidence import read,require
from .recoverable_model import VARIANT
from .colab_k7_validation import validate_k7_reference

def validate(root,run,approved_heads,authorization,identity):
    config=read(run/'configuration.json')
    require(config.get('sampling_variant')==VARIANT and config.get('not_original_frozen_trajectory') is True,'Missing Route B provenance')
    require(config.get('prospective_amendment')==authorization['scientific_decision_sha256'],'Amendment differs')
    require(config.get('continuation')==authorization,'Segment authorization differs')
    require(config['implementation_git_commit']==identity['head'],'Execution HEAD differs')
    result=validate_k7_reference(root,run,approved_heads)
    return dict(result,variant=VARIANT,trajectory_equivalence_to_original=False,scientific_amendment_sha256=authorization['scientific_decision_sha256'],registration_sha256=identity['registration_sha256'])
