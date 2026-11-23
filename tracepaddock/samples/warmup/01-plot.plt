// Synthetic warm-up: plotting and bit fields
TutorialSpeed nobit 5 unit=km/h scmin=25 scmax=65 fmt=.2f ref=50
TutorialTargetSpeed nobit 4 unit=km/h scmin=25 scmax=65 fmt=.0f
TutorialDriveMode nobit 3 scmin=-0.5 scmax=2.5
TutorialFlags nobit 7 scmin=0 scmax=120 fmt=8b
TutorialFlags:1 bit 6 key=TutorialLimitedFlag
TutorialFlags:2:2 nobit 8 key=TutorialModeBits scmin=0 scmax=4 fmt=.0f
~ TutorialFlags:4:3 nobit 9 key=TutorialTrimUnsigned scmin=-4 scmax=7 fmt=.0f
~ TutorialFlags:4:-3 nobit 10 key=TutorialTrimSigned scmin=-4 scmax=7 fmt=.0f
