import cv2
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import geocoder
from datetime import datetime
from ultralytics import YOLO
import threading
import os
import socket
import requests
import json
from math import radians, sin, cos, sqrt, atan2


class DisasterResponseSystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        #self.cap = cv2.VideoCapture('http://192.0.0.4:8080/video')
        #----------------------------------------------------------

        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("External camera not available. Falling back to default camera.")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Cannot open any camera. Please check your device connections.")

        print("Camera initialized successfully.")

            # Check if camera opened successfully
            #if not self.cap.isOpened():

                #print("Error: Could not open external camera. Falling back to default camera.")
                #self.cap = cv2.VideoCapture(0)  # Fall back to default camera

                # If still not working, raise an error
                #if not self.cap.isOpened():
                    #raise IOError("Cannot open any camera. Please check your connections.")
            #else:
               # print("External USB camera connected successfully!")

        #-----------------------------------------------------------
        #def find_available_camera(self):
           # """Scan for available cameras and return the first working index"""
            #for i in range(10):  # Try indices 0-9
             #   cap = cv2.VideoCapture(i)
              #  if cap.isOpened():
               #     cap.release()
                #    return i
           # return 0  # Return 0 if no cameras found
        #----------------------------------------------------------
        # self.cap = cv2.VideoCapture(1)
        self.db = self.initialize_firebase()
        self.sender_email = "mohamedasif7799@gmail.com"
        self.app_password = "sxwd fbpx vvkh alvz"
        self.receiver_email = "rmohamedasif1@gmail.com"
        self.emergency_contacts = [
            "rmohamedasif1@gmail.com",  # Primary contact
            "emergency@example.com"  # Secondary contact (replace with actual email)
        ]
        self.last_alert_time = 0
        self.alert_cooldown = 60  # 1 minute between alerts
        self.detection_threshold = 0.6
        self.current_emergency = "none"
        self.emergency_level = 0  # 0-none, 1-low, 2-medium, 3-high
        self.running = True
        self.drone_active = False
        self.drone_battery = 100
        self.previous_locations = []  # Store recent locations for accuracy
        self.location_accuracy = 0

        # Emergency severity thresholds
        self.fire_threshold = 0.05  # 5% of frame showing red
        self.flood_threshold = 0.08  # 8% of frame showing blue

        # Drone control parameters
        self.drone_dispatch_threshold = 2  # Minimum emergency level to dispatch drone
        self.drone_return_home = False

        # Create output directory for images
        os.makedirs("emergency_captures", exist_ok=True)

    def initialize_firebase(self):
        try:
            cred = credentials.Certificate("C:\\Users\\91934\\PycharmProjects\\drone1\\.venv\\Lib\\crd1.json")
            firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            return None

    def get_laptop_location(self):
        """Enhanced location detection with multiple sources and accuracy rating"""
        try:
            # Primary method: IP-based geocoding
            g = geocoder.ip('me')
            ip_location = {
                'latitude': g.latlng[0] if g.latlng else 0.0,
                'longitude': g.latlng[1] if g.latlng else 0.0,
                'address': g.address if g.address else 'Unknown',
                'source': 'ip'
            }

            # Secondary method: Using a web API for additional confirmation
            try:
                response = requests.get('https://ipinfo.io/json')
                if response.status_code == 200:
                    data = response.json()
                    if 'loc' in data:
                        lat, lng = map(float, data['loc'].split(','))
                        api_location = {
                            'latitude': lat,
                            'longitude': lng,
                            'address': data.get('city', '') + ', ' + data.get('region', '') + ', ' + data.get('country',
                                                                                                              ''),
                            'source': 'api'
                        }

                        # Calculate distance between two location sources
                        distance = self.calculate_distance(
                            ip_location['latitude'], ip_location['longitude'],
                            api_location['latitude'], api_location['longitude']
                        )

                        # If sources agree (within 10km), increase confidence
                        if distance < 10:
                            self.location_accuracy = min(self.location_accuracy + 10, 100)
                        else:
                            self.location_accuracy = max(self.location_accuracy - 5, 0)

                        # Use the average location for better accuracy
                        final_location = {
                            'latitude': (ip_location['latitude'] + api_location['latitude']) / 2,
                            'longitude': (ip_location['longitude'] + api_location['longitude']) / 2,
                            'address': ip_location['address'],  # Use the more detailed address
                            'accuracy': self.location_accuracy,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Add to location history (keep last 5)
                        self.previous_locations.append(final_location)
                        if len(self.previous_locations) > 5:
                            self.previous_locations.pop(0)

                        return final_location
            except Exception as e:
                print(f"Secondary location method failed: {e}")

            # If secondary method fails, use primary with lower confidence
            self.location_accuracy = max(self.location_accuracy - 10, 30)  # Decrease but maintain minimum confidence
            ip_location['accuracy'] = self.location_accuracy
            ip_location['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add to location history
            self.previous_locations.append(ip_location)
            if len(self.previous_locations) > 5:
                self.previous_locations.pop(0)

            return ip_location

        except Exception as e:
            print(f"Location detection error: {e}")
            # Use last known location if available
            if self.previous_locations:
                self.location_accuracy = max(self.location_accuracy - 20, 10)  # Significantly decrease confidence
                self.previous_locations[-1]['accuracy'] = self.location_accuracy
                return self.previous_locations[-1]

            return {
                'latitude': 0.0,
                'longitude': 0.0,
                'address': 'Unknown',
                'accuracy': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km using the Haversine formula"""
        R = 6371  # Earth radius in kilometers

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    def detect_humans(self, frame):
        results = self.model(frame)
        person_count = 0
        boxes = []

        for result in results:
            boxes_data = result.boxes
            for box in boxes_data:
                cls = int(box.cls.item())
                conf = box.conf.item()

                if cls == 0 and conf > self.detection_threshold:  # Person class is 0 in COCO dataset
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))

        return frame, person_count, boxes

    def save_image(self, frame, prefix="detected"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emergency_captures/{prefix}_{self.current_emergency}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def send_to_firebase(self, location_data, human_count, image_path):
        if not self.db:
            print("Firebase not initialized")
            return False

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            device_id = socket.gethostname()

            # Determine if drone is dispatched
            drone_dispatched = self.drone_active and self.emergency_level >= self.drone_dispatch_threshold

            doc_data = {
                'timestamp': timestamp,
                'location': {
                    'latitude': location_data['latitude'],
                    'longitude': location_data['longitude'],
                    'address': location_data['address'],
                    'accuracy': location_data.get('accuracy', 0)
                },
                'human_count': human_count,
                'image_path': image_path,
                'emergency_type': self.current_emergency,
                'emergency_level': self.emergency_level,
                'status': 'detected',
                'device_id': device_id,
                'system_type': 'laptop',
                'drone': {
                    'active': self.drone_active,
                    'dispatched': drone_dispatched,
                    'battery': self.drone_battery,
                    'status': 'en_route' if drone_dispatched else 'standby'
                }
            }

            self.db.collection('detections').add(doc_data)
            print(f"Data sent to Firebase - Emergency Level: {self.emergency_level}")
            return True
        except Exception as e:
            print(f"Firebase error: {e}")
            return False

    def send_email_alert(self, location_data, human_count, image_path):
        try:
            # Determine which contacts to notify based on emergency level
            recipients = [self.emergency_contacts[0]]  # Always notify primary contact
            if self.emergency_level >= 2:  # For medium and high emergencies, notify all contacts
                recipients = self.emergency_contacts

            message = MIMEMultipart()
            message["From"] = self.sender_email

            # Use different subjects based on emergency level
            urgency = "CRITICAL" if self.emergency_level == 3 else "URGENT" if self.emergency_level == 2 else "ALERT"
            message["Subject"] = f"{urgency}: {human_count} humans detected in {self.current_emergency} zone"

            # Create a more detailed email body
            body = f"""EMERGENCY {urgency}:

{human_count} human(s) detected in a {self.current_emergency.upper()} zone.
Emergency Level: {self.emergency_level}/3

Location:
- Latitude: {location_data['latitude']}
- Longitude: {location_data['longitude']}
- Address: {location_data['address']}
- Location Accuracy: {location_data.get('accuracy', 0)}%

Google Maps Link: https://www.google.com/maps?q={location_data['latitude']},{location_data['longitude']}

Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Drone Status: {"DISPATCHED" if (self.drone_active and self.emergency_level >= self.drone_dispatch_threshold) else "On Standby"}
Drone Battery: {self.drone_battery}%

{"IMMEDIATE ACTION REQUIRED" if self.emergency_level >= 2 else "Please dispatch emergency services to this location."}

This is an automated alert from the Disaster Response System."""

            message.attach(MIMEText(body, "plain"))

            with open(image_path, "rb") as file:
                image = MIMEImage(file.read())
                image.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
                message.attach(image)

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.sender_email, self.app_password)

            # Send to all recipients
            for recipient in recipients:
                message["To"] = recipient
                server.send_message(message)

            server.quit()
            print(f"Email alert sent to {len(recipients)} recipients")
            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False

    def process_alerts(self, frame, human_count):
        current_time = time.time()

        # Determine if we should send an alert based on time elapsed and emergency level
        should_alert = False

        # For high emergencies (level 3), reduce cooldown to 30 seconds
        if self.emergency_level == 3 and (current_time - self.last_alert_time) > 30:
            should_alert = True
        # For medium emergencies (level 2), use standard cooldown
        elif self.emergency_level == 2 and (current_time - self.last_alert_time) > self.alert_cooldown:
            should_alert = True
        # For low emergencies (level 1), only alert if humans are detected and standard cooldown
        elif self.emergency_level == 1 and human_count > 0 and (
                current_time - self.last_alert_time) > self.alert_cooldown:
            should_alert = True

        if should_alert:
            self.last_alert_time = current_time

            location_data = self.get_laptop_location()
            image_path = self.save_image(frame, f"emergency_lvl{self.emergency_level}")

            firebase_thread = threading.Thread(target=self.send_to_firebase,
                                               args=(location_data, human_count, image_path))
            email_thread = threading.Thread(target=self.send_email_alert, args=(location_data, human_count, image_path))

            firebase_thread.start()
            email_thread.start()

            # If drone is active and emergency level is high enough, dispatch drone
            if self.drone_active and self.emergency_level >= self.drone_dispatch_threshold:
                self.dispatch_drone(location_data)

    def detect_emergency_type(self, frame):
        """Enhanced emergency detection with severity levels"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        total_pixels = height * width

        # Fire detection (red)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = red_mask1 + red_mask2
        red_pixels = np.sum(red_mask > 0)
        red_percentage = red_pixels / total_pixels

        # Flood detection (blue)
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = np.sum(blue_mask > 0)
        blue_percentage = blue_pixels / total_pixels

        # Smoke detection (grey/white)
        lower_smoke = np.array([0, 0, 150])
        upper_smoke = np.array([180, 30, 255])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        smoke_pixels = np.sum(smoke_mask > 0)
        smoke_percentage = smoke_pixels / total_pixels

        # Determine emergency type and level based on color percentages
        previous_emergency = self.current_emergency
        previous_level = self.emergency_level

        # Reset emergency status
        self.current_emergency = "none"
        self.emergency_level = 0

        # Check for fire (highest priority)
        if red_percentage > self.fire_threshold:
            self.current_emergency = "fire"
            # Determine severity
            if red_percentage > self.fire_threshold * 4:  # More than 20% of frame is red
                self.emergency_level = 3  # High emergency
            elif red_percentage > self.fire_threshold * 2:  # More than 10% of frame is red
                self.emergency_level = 2  # Medium emergency
            else:
                self.emergency_level = 1  # Low emergency

            # Check for smoke with fire for higher severity
            if smoke_percentage > 0.15 and self.emergency_level < 3:
                self.emergency_level = 3  # Upgrade to high emergency if significant smoke

        # Check for flood (second priority)
        elif blue_percentage > self.flood_threshold:
            self.current_emergency = "flood"
            # Determine severity
            if blue_percentage > self.flood_threshold * 3:  # More than 24% of frame is blue
                self.emergency_level = 3  # High emergency
            elif blue_percentage > self.flood_threshold * 1.5:  # More than 12% of frame is blue
                self.emergency_level = 2  # Medium emergency
            else:
                self.emergency_level = 1  # Low emergency

        # Check for smoke only (lowest priority)
        elif smoke_percentage > 0.2:  # 20% of frame is smoke/grey
            self.current_emergency = "smoke"
            self.emergency_level = 1  # Start with low emergency
            if smoke_percentage > 0.4:  # 40% of frame is smoke
                self.emergency_level = 2  # Medium emergency

        # Visual indicators on frame
        emergency_color = (0, 0, 255)  # Red for default
        if self.current_emergency == "fire":
            emergency_color = (0, 0, 255)  # Red for fire
        elif self.current_emergency == "flood":
            emergency_color = (255, 0, 0)  # Blue for flood
        elif self.current_emergency == "smoke":
            emergency_color = (128, 128, 128)  # Grey for smoke

        # Draw emergency status indicators
        cv2.putText(frame, f"Emergency: {self.current_emergency.upper()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, emergency_color, 2)
        cv2.putText(frame, f"Level: {self.emergency_level}/3", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, emergency_color, 2)

        # If emergency type or level changed, log it
        if previous_emergency != self.current_emergency or previous_level != self.emergency_level:
            print(f"Emergency update: {self.current_emergency} (Level {self.emergency_level})")

            # If transitioning to a higher emergency level, save image
            if self.emergency_level > previous_level and self.emergency_level >= 2:
                self.save_image(frame, f"transition_to_level{self.emergency_level}")

    def dispatch_drone(self, location_data):
        """Simulate drone dispatch to emergency location"""
        if not self.drone_active:
            print("Drone not active, cannot dispatch")
            return

        print(f"Dispatching drone to location: {location_data['latitude']}, {location_data['longitude']}")

        # Simulate drone activity with a separate thread
        drone_thread = threading.Thread(target=self.simulate_drone_mission, args=(location_data,))
        drone_thread.daemon = True
        drone_thread.start()

        # Update Firebase with drone dispatch status
        if self.db:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                drone_status = {
                    'status': 'dispatched',
                    'mission_start': timestamp,
                    'target_location': {
                        'latitude': location_data['latitude'],
                        'longitude': location_data['longitude'],
                        'address': location_data['address']
                    },
                    'battery': self.drone_battery,
                    'emergency_type': self.current_emergency,
                    'emergency_level': self.emergency_level
                }

                self.db.collection('drone_missions').add(drone_status)
            except Exception as e:
                print(f"Failed to update drone status in Firebase: {e}")

    def simulate_drone_mission(self, location_data):
        """Simulate a drone mission to the emergency location"""
        print("Drone mission started")

        # Simulate travel time (5 seconds)
        time.sleep(5)

        # Simulate arrival at location
        print(f"Drone arrived at {location_data['address']}")
        self.drone_battery -= 10  # Reduce battery level

        # Simulate scanning (5 seconds)
        time.sleep(5)

        # Simulate data collection
        drone_findings = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'location': {
                'latitude': location_data['latitude'],
                'longitude': location_data['longitude'],
                'address': location_data['address']
            },
            'emergency_confirmed': True,
            'emergency_type': self.current_emergency,
            'severity_assessment': self.emergency_level,
            'human_detection': {
                'humans_detected': True,
                'estimated_count': 2,  # Simulated count
                'requires_evacuation': self.emergency_level >= 2
            },
            'battery_remaining': self.drone_battery,
            'recommendations': [
                "Dispatch emergency services immediately",
                "Set up evacuation corridor" if self.emergency_level >= 2 else "Monitor situation"
            ]
        }

        # Update Firebase with drone findings
        if self.db:
            try:
                self.db.collection('drone_findings').add(drone_findings)
                print("Drone findings uploaded to Firebase")
            except Exception as e:
                print(f"Failed to upload drone findings: {e}")

        # Simulate return journey
        time.sleep(5)
        self.drone_battery -= 10  # Reduce battery level
        print("Drone returned to base")

        # Check if battery needs charging
        if self.drone_battery < 30:
            print("Drone battery low, charging required")
            self.drone_return_home = True

    def prepare_for_drone_integration(self):
        """Initialize drone systems and prepare for potential dispatch"""
        print("System ready for drone integration")
        print("Current setup: Using laptop camera and enhanced GPS")
        print("Attempting to activate drone systems...")

        try:
            # Simulate drone activation
            time.sleep(2)  # Simulate connection time
            self.drone_active = True
            self.drone_battery = 95  # Simulate partial charge

            drone_status = {
                'status': 'connected',
                'ready_for_dispatch': True,
                'battery': self.drone_battery,
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'capabilities': ['live_video', 'thermal_imaging', 'gps_navigation']
            }

            print("Drone systems activated and ready for dispatch")
            print(f"Drone battery level: {self.drone_battery}%")

            if self.db:
                try:
                    self.db.collection('system_status').document('drone_status').set(drone_status)
                    print("Drone status updated in Firebase")
                except Exception as e:
                    print(f"Could not update drone status in Firebase: {e}")

        except Exception as e:
            print(f"Drone activation failed: {e}")
            self.drone_active = False

            if self.db:
                self.db.collection('system_status').document('drone_status').set({
                    'status': 'failed_to_connect',
                    'error': str(e),
                    'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    def monitor_drone_battery(self):
        """Monitor drone battery and simulate usage"""
        while self.running and self.drone_active:
            time.sleep(60)  # Check every minute

            # Simulate battery drain
            if not self.drone_return_home:
                self.drone_battery = max(0, self.drone_battery - 1)
            else:
                # Simulate charging
                self.drone_battery = min(100, self.drone_battery + 5)
                if self.drone_battery >= 90:
                    self.drone_return_home = False
                    print("Drone charged and ready for new missions")

            # Update Firebase with current battery status
            if self.db:
                try:
                    self.db.collection('system_status').document('drone_status').update({
                        'battery': self.drone_battery,
                        'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception:
                    pass  # Ignore update errors

    def run(self):
        frame_count = 0
        start_time = time.time()
        fps_display = 0

        self.prepare_for_drone_integration()

        # Start battery monitoring in a separate thread
        if self.drone_active:
            battery_thread = threading.Thread(target=self.monitor_drone_battery)
            battery_thread.daemon = True
            battery_thread.start()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break

            frame_count += 1

            processed_frame, human_count, boxes = self.detect_humans(frame)

            self.detect_emergency_type(processed_frame)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps_display = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            fps_text = f"FPS: {fps_display:.1f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            count_text = f"Persons: {human_count}"
            cv2.putText(processed_frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            location = self.get_laptop_location()
            location_text = f"GPS: {location['latitude']:.5f}, {location['longitude']:.5f} (±{location.get('accuracy', 0)}%)"
            cv2.putText(processed_frame, location_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            system_status = "Laptop + Drone" if self.drone_active else "Laptop Only"
            if self.drone_active:
                drone_text = f"Drone: {'DISPATCHED' if (self.emergency_level >= self.drone_dispatch_threshold and self.current_emergency != 'none') else 'Ready'} | Batt: {self.drone_battery}%"
                cv2.putText(processed_frame, drone_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            system_text = f"System: {system_status}"
            cv2.putText(processed_frame, system_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Disaster Response System", processed_frame)

            self.process_alerts(processed_frame, human_count)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")


if __name__ == "__main__":
    system = DisasterResponseSystem()
    system.run()
