angles = [
('SpineBase', 'SpineMid', 'SpineShoulder'),
('SpineMid', 'SpineShoulder', 'Neck'),
('SpineBase', 'SpineShoulder', 'Neck'),
('HipLeft', 'SpineBase', 'SpineMid'),
('HipLeft', 'SpineBase', 'Neck'),
('HipRight', 'SpineBase', 'SpineMid'),
('HipRight', 'SpineBase', 'Neck'),
('AnkleRight', 'KneeRight', 'HipRight'),
('AnkleRight', 'HipRight', 'SpineBase'),
('KneeRight', 'HipRight', 'SpineBase'),
('AnkleLeft', 'KneeLeft', 'HipLeft'),
('AnkleLeft', 'HipLeft', 'SpineBase'),
('KneeLeft', 'HipLeft', 'SpineBase')
]

# ('WristLeft', 'ElbowLeft', 'ShoulderLeft'),
# ('WristLeft', 'ShoulderLeft', 'SpineShoulder'),
# ('ShoulderLeft', 'SpineShoulder', 'Neck'),
# ('WristRight', 'ElbowRight', 'ShoulderRight'),
# ('WristRight', 'ShoulderRight', 'SpineShoulder'),
# ('ShoulderRight', 'SpineShoulder', 'Neck')]