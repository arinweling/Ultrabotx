import genesis as gs
print("SENSORS:", dir(gs.sensors))
try:
    print("CAMERA:", dir(gs.sensors.Camera))
except Exception as e:
    print(e)
