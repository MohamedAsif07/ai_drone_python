# Complete Step-by-Step Guide: Connecting Disaster Response Code to Your Drone

## Phase 1: Hardware Setup

### Step 1: Prepare Your Drone Hardware
1. **Flight Controller**: Ensure you have a compatible flight controller (Pixhawk, APM, etc.)
2. **Telemetry Radio**: 
   - 433MHz or 915MHz telemetry radio pair
   - One connects to drone, one to ground station (laptop)
3. **Camera**: 
   - USB camera connected to companion computer (Raspberry Pi) OR
   - Direct USB camera connection to laptop for ground testing
4. **Companion Computer** (Optional but recommended):
   - Raspberry Pi 4 with camera module
   - Connected to flight controller via UART/Serial

### Step 2: Physical Connections
```
Drone Side:
Flight Controller ←→ Telemetry Radio ←→ [Air]
Flight Controller ←→ Companion Computer (if using)

Ground Side:
Laptop ←→ USB Telemetry Radio ←→ [Air]
Laptop ←→ USB Camera (if not using companion computer)
```

## Phase 2: Software Installation

### Step 3: Install Mission Planner
1. Download Mission Planner from: https://ardupilot.org/planner/
2. Install and run Mission Planner
3. Connect your drone via USB first to configure

### Step 4: Install Python Dependencies
Open Command Prompt/Terminal as Administrator:
```bash
# Install Python packages
pip install pymavlink
pip install ultralytics
pip install opencv-python
pip install firebase-admin
pip install geocoder
pip install numpy
pip install requests

# For Windows users, if you get errors:
pip install --upgrade pip
pip install pymavlink --no-cache-dir
```

### Step 5: Download YOLO Model
```bash
# Create project directory
mkdir drone_disaster_system
cd drone_disaster_system

# The YOLO model will auto-download on first run
# Or manually download:
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Phase 3: Mission Planner Configuration

### Step 6: Configure Mission Planner for MAVLink Output
1. **Connect drone to Mission Planner**:
   - Connect via USB cable
   - Select correct COM port
   - Set baud rate to 115200
   - Click "Connect"

2. **Enable MAVLink Output**:
   - Go to `CONFIG/TUNING` → `Full Parameter List`
   - Find `SERIAL1_PROTOCOL` and set to `2` (MAVLink2)
   - Find `SERIAL1_BAUD` and set to `57` (57600 baud)
   - Write parameters to drone

3. **Setup Telemetry Output**:
   - Go to `SETUP` → `Optional Hardware` → `MAVLink`
   - Enable UDP connection on port 14550
   - Set IP to 127.0.0.1 for local connection

### Step 7: Test MAVLink Connection
1. **In Mission Planner**:
   - Go to `DATA` screen
   - Verify you're receiving telemetry data
   - Note GPS coordinates, battery voltage, flight mode

2. **Test MAVLink Stream**:
   ```python
   # Test script - save as test_mavlink.py
   from pymavlink import mavutil
   
   # Connect to Mission Planner
   master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
   
   # Wait for heartbeat
   master.wait_heartbeat()
   print("Heartbeat received!")
   
   # Get GPS position
   while True:
       msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
       if msg:
           print(f"GPS: {msg.lat/1e7}, {msg.lon/1e7}, Alt: {msg.alt/1000}m")
           break
   ```

## Phase 4: Firebase Setup

### Step 8: Create Firebase Project
1. Go to https://console.firebase.google.com/
2. Create new project: "drone-disaster-response"
3. Enable Firestore Database
4. Go to Project Settings → Service Accounts
5. Generate new private key → Download JSON file
6. Rename to `firebase_credentials.json`
7. Place in your project directory

### Step 9: Test Firebase Connection
```python
# Test script - save as test_firebase.py
import firebase_admin
from firebase_admin import credentials, firestore

try:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # Test write
    db.collection('test').add({'message': 'Firebase connected!'})
    print("Firebase connection successful!")
except Exception as e:
    print(f"Firebase error: {e}")
```

## Phase 5: Email Configuration

### Step 10: Setup Gmail for Alerts
1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
   - Copy the 16-character password

3. **Update Code**:
   ```python
   self.sender_email = "your_email@gmail.com"
   self.app_password = "your_16_char_app_password"
   self.receiver_email = "emergency_contact@gmail.com"
   ```

## Phase 6: Camera Setup

### Step 11: Camera Configuration
1. **Test Camera**:
   ```python
   import cv2
   
   # Test different camera indices
   for i in range(5):
       cap = cv2.VideoCapture(i)
       if cap.isOpened():
           ret, frame = cap.read()
           if ret:
               print(f"Camera {i} working!")
               cv2.imshow(f"Camera {i}", frame)
               cv2.waitKey(1000)
           cap.release()
       cv2.destroyAllWindows()
   ```

2. **Note working camera index** (usually 0 or 1)

## Phase 7: Running the System

### Step 12: File Structure Setup
Create this directory structure:
```
drone_disaster_system/
├── drone_disaster_system.py
├── firebase_credentials.json
├── test_mavlink.py
├── test_firebase.py
├── drone_captures/          (auto-created)
├── mission_logs/           (auto-created)
└── yolov8n.pt             (auto-downloaded)
```

### Step 13: Pre-flight Testing
1. **Test with Mission Planner SITL**:
   ```bash
   # Start SITL in Mission Planner first
   python drone_disaster_system.py --connect udp:127.0.0.1:14550
   ```

2. **Verify all systems**:
   - Camera feed appears
   - GPS coordinates displayed
   - Emergency detection working
   - Firebase logging active

### Step 14: Live Drone Connection

#### Option A: USB Connection (Ground Testing)
```bash
python drone_disaster_system.py --connect COM3  # Replace COM3 with your port
```

#### Option B: Telemetry Radio Connection
1. **Connect telemetry radio to laptop**
2. **Check COM port in Device Manager**
3. **Run system**:
```bash
python drone_disaster_system.py --connect COM5  # Replace with telemetry COM port
```

#### Option C: WiFi/Network Connection
```bash
python drone_disaster_system.py --connect udp:192.168.1.100:14550
```

### Step 15: Flight Operations

#### Pre-Flight Checklist:
- [ ] Drone battery charged
- [ ] Telemetry radio connected
- [ ] Camera working
- [ ] GPS lock acquired
- [ ] Firebase connection tested
- [ ] Email alerts configured
- [ ] Emergency contacts notified

#### During Flight:
1. **Monitor system status**:
   - GPS coordinates updating
   - Battery percentage
   - Flight mode display
   - Emergency detection active

2. **Emergency Controls**:
   - Press 'r' for Return to Launch
   - Press 'q' to quit system
   - System auto-RTL on critical emergencies

## Phase 8: Troubleshooting

### Common Issues and Solutions:

#### Connection Issues:
```bash
# Check COM ports
python -c "import serial.tools.list_ports; print([port.device for port in serial.tools.list_ports.comports()])"

# Test MAVLink connection
python -c "from pymavlink import mavutil; m=mavutil.mavlink_connection('COM3'); m.wait_heartbeat(); print('Connected!')"
```

#### Camera Issues:
```bash
# List available cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

#### Permission Issues (Linux):
```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB0
```

### Error Messages and Fixes:

| Error | Solution |
|-------|----------|
| "No module named pymavlink" | `pip install pymavlink` |
| "Cannot open camera" | Check camera index, try different values |
| "Connection refused" | Check Mission Planner MAVLink output settings |
| "Firebase permission denied" | Check credentials file path and permissions |
| "SMTP authentication failed" | Use Gmail app password, not regular password |

## Phase 9: Advanced Configuration

### Step 16: Mission Planning Integration
```python
# Add to your code for waypoint missions
def upload_mission(self, waypoints):
    if not self.vehicle:
        return False
        
    # Clear existing mission
    self.vehicle.mav.mission_clear_all_send(
        self.vehicle.target_system,
        self.vehicle.target_component
    )
    
    # Upload new waypoints
    for i, wp in enumerate(waypoints):
        self.vehicle.mav.mission_item_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            i,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1,  # current, autocontinue
            0, 0, 0, 0,  # param1-4
            wp['lat'], wp['lon'], wp['alt']
        )
```

### Step 17: Real-time Streaming (Optional)
For live video streaming to ground station:
```bash
# On drone (Raspberry Pi)
raspivid -t 0 -w 640 -h 480 -fps 30 -b 1000000 -o - | nc -l 8080

# On ground station
nc drone_ip 8080 | mplayer -fps 30 -cache 1024 -
```

## Safety Considerations

⚠️ **IMPORTANT SAFETY NOTES**:

1. **Always test thoroughly** in SITL before real flight
2. **Keep manual control** - be ready to switch to manual mode
3. **Monitor battery levels** - land before critical battery
4. **Follow local regulations** - check drone laws in your area
5. **Have emergency procedures** - know how to emergency land
6. **Test range limits** - ensure telemetry range is adequate
7. **Weather conditions** - only fly in suitable weather

## Support Commands

### Quick Status Check:
```bash
# Check all connections
python -c "
import cv2, serial.tools.list_ports, firebase_admin
from pymavlink import mavutil

print('Cameras:', [i for i in range(5) if cv2.VideoCapture(i).isOpened()])
print('Serial ports:', [p.device for p in serial.tools.list_ports.comports()])
print('Testing MAVLink...', end='')
try:
    m = mavutil.mavlink_connection('udp:127.0.0.1:14550', timeout=5)
    m.wait_heartbeat()
    print('OK')
except:
    print('FAILED')
"
```

This guide should get your drone disaster response system fully operational. Start with ground testing and SITL simulation before attempting actual flights!
