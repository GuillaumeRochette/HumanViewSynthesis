JOINTS = {
    # Body.
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "Sternum": 17,
    "Head": 18,
    # Left Hand.
    # Thumb.
    "LThumb1CMC": 19,
    "LThumb2Knuckles": 20,
    "LThumb3IP": 21,
    "LThumb4FingerTip": 22,
    # Index.
    "LIndex1Knuckles": 23,
    "LIndex2PIP": 24,
    "LIndex3DIP": 25,
    "LIndex4FingerTip": 26,
    # Middle.
    "LMiddle1Knuckles": 27,
    "LMiddle2PIP": 28,
    "LMiddle3DIP": 29,
    "LMiddle4FingerTip": 30,
    # Ring.
    "LRing1Knuckles": 31,
    "LRing2PIP": 32,
    "LRing3DIP": 33,
    "LRing4FingerTip": 34,
    # Pinky.
    "LPinky1Knuckles": 35,
    "LPinky2PIP": 36,
    "LPinky3DIP": 37,
    "LPinky4FingerTip": 38,
    # Right Hand.
    # Thumb.
    "RThumb1CMC": 39,
    "RThumb2Knuckles": 40,
    "RThumb3IP": 41,
    "RThumb4FingerTip": 42,
    # Index.
    "RIndex1Knuckles": 43,
    "RIndex2PIP": 44,
    "RIndex3DIP": 45,
    "RIndex4FingerTip": 46,
    # Middle.
    "RMiddle1Knuckles": 47,
    "RMiddle2PIP": 48,
    "RMiddle3DIP": 49,
    "RMiddle4FingerTip": 50,
    # Ring.
    "RRing1Knuckles": 51,
    "RRing2PIP": 52,
    "RRing3DIP": 53,
    "RRing4FingerTip": 54,
    # Pinky.
    "RPinky1Knuckles": 55,
    "RPinky2PIP": 56,
    "RPinky3DIP": 57,
    "RPinky4FingerTip": 58,
    # Face.
    # Contour.
    "FaceContour5": 59,
    "FaceContour6": 60,
    "FaceContour7": 61,
    "FaceContour8": 62,
    "FaceContour9": 63,
    "FaceContour10": 64,
    "FaceContour11": 65,
    # Right Eyebrow.
    "REyeBrow0": 66,
    "REyeBrow1": 67,
    "REyeBrow2": 68,
    "REyeBrow3": 69,
    "REyeBrow4": 70,
    # Left Eyebrow.
    "LEyeBrow4": 71,
    "LEyeBrow3": 72,
    "LEyeBrow2": 73,
    "LEyeBrow1": 74,
    "LEyeBrow0": 75,
    # Upper Nose.
    "NoseUpper0": 76,
    "NoseUpper1": 77,
    "NoseUpper2": 78,
    "NoseUpper3": 79,
    # Lower Nose.
    "NoseLower0": 80,
    "NoseLower1": 81,
    "NoseLower2": 82,
    "NoseLower3": 83,
    "NoseLower4": 84,
    # Right Eye.
    "REye0": 85,
    "REye1": 86,
    "REye2": 87,
    "REye3": 88,
    "REye4": 89,
    "REye5": 90,
    # Left Eye.
    "LEye0": 91,
    "LEye1": 92,
    "LEye2": 93,
    "LEye3": 94,
    "LEye4": 95,
    "LEye5": 96,
    # Outer Mouth.
    "OMouth0": 97,
    "OMouth1": 98,
    "OMouth2": 99,
    "OMouth3": 100,
    "OMouth4": 101,
    "OMouth5": 102,
    "OMouth6": 103,
    "OMouth7": 104,
    "OMouth8": 105,
    "OMouth9": 106,
    "OMouth10": 107,
    "OMouth11": 108,
    # Inner Mouth.
    "IMouth0": 109,
    "IMouth1": 110,
    "IMouth2": 111,
    "IMouth3": 112,
    "IMouth4": 113,
    "IMouth5": 114,
    "IMouth6": 115,
    "IMouth7": 116,
}

EDGES = (
    # Body.
    (18, 17),
    (17, 5),
    (17, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (17, 11),
    (17, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (11, 12),
    # Left Hand.
    # Thumb.
    (9, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    # Index.
    (9, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    # Middle.
    (9, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    # Ring.
    (9, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    # Pinky.
    (9, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    # Right Hand.
    # Thumb.
    (10, 39),
    (39, 40),
    (40, 41),
    (41, 42),
    # Index.
    (10, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    # Middle.
    (10, 47),
    (47, 48),
    (48, 49),
    (49, 50),
    # Ring.
    (10, 51),
    (51, 52),
    (52, 53),
    (53, 54),
    # Pinky.
    (10, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    # Face.
    # Contour.
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    # Contour to Ears.
    (4, 59),
    (3, 65),
    # Right Eyebrow.
    (66, 67),
    (67, 68),
    (68, 69),
    (69, 70),
    # Left Eyebrow.
    (71, 72),
    (72, 73),
    (73, 74),
    (74, 75),
    # Upper Nose.
    (76, 77),
    (77, 78),
    (78, 79),
    # Lower Nose.
    (80, 81),
    (81, 82),
    (82, 83),
    (83, 84),
    # Right Eye.
    (85, 86),
    (86, 87),
    (87, 88),
    (88, 89),
    (89, 90),
    (90, 85),
    # Left Eye.
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    (95, 96),
    (96, 91),
    # Outer Mouth.
    (97, 98),
    (98, 99),
    (99, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    (103, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    (107, 108),
    # Inner Mouth.
    (109, 110),
    (110, 111),
    (111, 112),
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    (116, 109),
)

BODY_135_TO_BODY_117 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
]