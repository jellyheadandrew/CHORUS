EPSILON_DEFAULT = 0.13 # interaction-distance threshold
ALL_BODY_SEGMENTATION_NAMES = [ # 24 different body parts
    'rightHand', 
    'rightUpLeg', 
    'leftArm', 
    'leftLeg', 
    'leftToeBase', 
    'leftFoot', 
    'spine1', 
    'spine2', 
    'leftShoulder', 
    'rightShoulder', 
    'rightFoot', 
    'head', 
    'rightArm', 
    'leftHandIndex1', 
    'rightLeg', 
    'rightHandIndex1', 
    'leftForeArm', 
    'rightForeArm', 
    'neck', 
    'rightToeBase', 
    'spine', 
    'leftUpLeg', 
    'leftHand', 
    'hips'
]
BODY_PART_CORRESPONDENCE = { # key: 'body_part_define_method'
    # quant (dummy)
    'quant': {
        'fullbody': {
            'correspondence_map': {
                'pass': [
                    ALL_BODY_SEGMENTATION_NAMES,
                ],
                'block': [
                ]
            },
            'interaction_threshold_per_world': {'SMPL': EPSILON_DEFAULT}
        },        
    },
    # demo
    'qual:demo': {
        'rightArm': {
            'correspondence_map': {
                'pass': [
                    ['rightArm', 'rightForeArm'],
                ],
                'block': [
                ]
            },
            'interaction_threshold_per_world': {'SMPL': EPSILON_DEFAULT}
        },
        'leftFoot': {
            'correspondence_map': {
                'pass': [
                    ['leftFoot', 'leftToeBase'],
                ],
                'block': [
                ]
            },
            'interaction_threshold_per_world': {'SMPL': EPSILON_DEFAULT}
        },
        'torso': {
            'correspondence_map': {
                'pass': [
                    ['spine', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder'],
                ],
                'block': [
                ]
            },
            'interaction_threshold_per_world': {'SMPL': EPSILON_DEFAULT}
        },
    },
}