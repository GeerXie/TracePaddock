// Synthetic warm-up: apply operations to copies
TutorialNoisySpeed nobit 8 scmin=0 scmax=22
TutorialNoisySpeed nobit 7 key=TutorialFilteredSpeed scmin=0 scmax=22 dgf=bth:3:2
~ TutorialNoisySpeed nobit 6 key=TutorialAverageSpeed scmin=0 scmax=22 mavr(-10)
TutorialSpeedBaseline nobit 4 scmin=0 scmax=22
~ TutorialSpeedBaseline nobit 3 key=TutorialSpeedKmh unit=km/h scmin=0 scmax=80 exp=3.6*x
TutorialSpeedBaseline nobit 9 key=TutorialDerivative unit=m/s2 scmin=-2 scmax=3 diff
~ TutorialSpeedBaseline nobit 10 key=TutorialDistance unit=m scmin=0 scmax=210 integ
~ TutorialSpeedBaseline nobit 11 key=TutorialSpeedCentered scmin=-5 scmax=5 -mean
TutorialAcceleration nobit 5 scmin=-2 scmax=3
