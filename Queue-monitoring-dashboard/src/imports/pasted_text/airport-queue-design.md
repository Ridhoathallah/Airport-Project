Evolve the existing dashboard design rather than rebuilding it from scratch.

Official project name:

Airport X-Ray Queue Monitoring System

The current implementation is functionally strong and the information architecture is already approved. Keep the existing functionality, data relationships, monitoring sections, and overall layout unless a change is necessary to improve usability.

The main goal of this iteration is VISUAL AND UX EVOLUTION.

The current design feels too much like an "AI dashboard" or futuristic computer-vision demo.

Make it feel like a real airport security operations monitoring system.

The technology is AI-powered internally, but the operator should experience this as an operational airport monitoring tool.

---

## 1. PRODUCT POSITIONING

Use the official name everywhere:

Airport X-Ray Queue Monitoring System

Do not rename or rebrand the application.

Optional contextual subtitle:

Terminal 1 · Security Screening

Do NOT use branding such as:

- AI Queue Monitor
- AI Dashboard
- Computer Vision Dashboard
- YOLO Monitor
- Smart AI Screening

YOLOv5 and SORT are implementation technologies, not the product identity.

---

## 2. DESIGN PHILOSOPHY

Change the visual direction from:

"AI / futuristic / technical showcase"

to:

"Airport operations / security monitoring / professional control room"

The dashboard should feel believable as software used continuously by airport/security operations staff.

Prioritize:

- clarity
- trust
- operational awareness
- fast scanning
- readable numbers
- restrained visual hierarchy
- practical information

Avoid:

- neon effects
- excessive cyan/purple
- futuristic HUD styling
- excessive glow
- excessive gradients
- glassmorphism
- overly rounded cards
- decorative AI graphics
- excessive technical labels
- visual effects that compete with operational information

The design should be sophisticated because of its information architecture and typography, not because of flashy effects.

---

## 3. KEEP THE APPROVED INFORMATION ARCHITECTURE

Keep the current structure:

Top status bar
→ main monitoring area
→ queue status
→ live camera
→ KPIs
→ traffic/queue analytics
→ waiting time
→ system health
→ recent activity/event timeline
→ diagnostics drawer

The existing 1440×900 desktop control-room layout remains the primary target.

Do not introduce unnecessary pages.

This is primarily a single-screen monitoring console.

---

## 4. INFORMATION PRIORITY

The operator should understand the situation within approximately 3–5 seconds.

Priority order:

1. Overall queue condition
2. People currently waiting
3. Occupancy/capacity
4. Queue direction: increasing / stable / clearing
5. Waiting time
6. Live camera
7. Recent operational activity
8. System/camera health
9. Technical diagnostics

The dashboard should answer:

"Is everything okay?"

"How many people are waiting?"

"Is the queue getting worse or better?"

"How long are people waiting?"

"Does the camera provide a reliable view?"

"Does the operator need to pay attention?"

---

## 5. QUEUE STATUS

Keep Queue Status as the dominant operational component.

Example:

QUEUE STATUS

BUSY

18 passengers waiting

60% capacity

↑ Queue increasing

Average wait
04:32

If congestion occurs:

CONGESTED

24 passengers waiting

80% capacity

↑ Queue increasing

Congestion
03:42

Reason:
High occupancy and low exit flow

Use restrained status colors:

NORMAL → green
BUSY → amber
CONGESTED → orange/red
BACKLOG → red
DEGRADED → amber
OFFLINE → muted red/dark

Do not rely on color alone.

Always combine:

color + label + icon/text

Avoid excessive glow around status elements.

---

## 6. LIVE CAMERA

The camera is an important part of the product because this is a camera-only system.

Make the CCTV feed feel like a real airport security camera rather than an AI demonstration.

Use:

LIVE CAMERA

CAM-01
Security Checkpoint A

Show:

- realistic CCTV-style video frame
- subtle timestamp
- camera identifier
- small LIVE indicator
- optional FPS indicator
- subtle person bounding boxes
- anonymous tracking IDs
- queue ROI
- entry/exit line

The computer-vision overlay should be subtle.

The camera should still visually read as:

"airport CCTV monitoring"

rather than:

"AI object detection demo."

Do not cover people with excessive boxes, labels, confidence scores, or graphics.

Do not display large:

YOLO
AI DETECTION
COMPUTER VISION
INFERENCE

labels over the camera.

If technical information is needed, keep it inside diagnostics.

---

## 7. YOLO / SORT

The project uses YOLOv5 for person detection and SORT for tracking.

Do not remove these technologies from the system.

Simply move them to the appropriate technical context.

Operator-facing:

Camera
● Live

Queue Monitoring
● Active

Technician-facing diagnostics:

Detection Model
YOLOv5s

Tracking
SORT

Inference
24.8 ms

Device
GPU

This separation is important.

The system should use AI without looking like an "AI product."

---

## 8. KPI DESIGN

Use practical operational terminology.

Prefer:

PEOPLE WAITING
18

CAPACITY
30

OCCUPANCY
60%

ENTRY RATE
12 / min

EXIT RATE
8 / min

AVERAGE WAIT
04:32

LONGEST WAIT
08:47

THROUGHPUT
120 / hour

NET QUEUE CHANGE
+4

Avoid technical KPI labels such as:

- AI detections
- tracked objects
- CV confidence
- inference objects
- model performance

unless they belong in diagnostics.

Numbers should be highly readable.

Use tabular/monospace numerals where appropriate so changing values do not visually jump.

---

## 9. TRAFFIC ANALYTICS

Keep the existing charts but make them feel like operational analytics rather than AI analytics.

Section:

QUEUE & TRAFFIC

Show:

Entries
Exits
Queue

Allow:

5m · 15m · 30m · 60m

The chart should make the queue direction immediately understandable.

Use subtle colors:

Entries → blue
Exits → aqua/green-blue
Queue → neutral/light blue

Do not use excessive saturated colors.

---

## 10. WAITING TIME

Section:

WAITING TIME

Show:

Average
04:32

Longest
08:47

Include a clean historical line chart.

If a threshold is exceeded, use a subtle warning marker.

Avoid large alert graphics.

---

## 11. OCCUPANCY

Use:

QUEUE OCCUPANCY

18 / 30

60%

Show a clean horizontal capacity meter or simple visualization.

Avoid large donut charts and unnecessary gauges.

The relationship between:

people
capacity
percentage

must be immediately understandable.

---

## 12. RECENT ACTIVITY

Rename technical event terminology into operator-friendly language.

Use:

RECENT ACTIVITY

Examples:

14:31:58
Passenger entered queue
18 → 19

14:30:42
Queue status changed
NORMAL → BUSY

14:29:11
Passenger cleared screening
17 → 16

14:27:54
Queue beginning to clear

Do not show implementation names such as:

track_enter_roi
track_lost
state_change
congestion_start

Those are engineering events, not operator events.

---

## 13. SYSTEM HEALTH

Keep critical health visible.

However, make it operational instead of overly technical.

Top-level:

CAMERA
● LIVE

QUEUE MONITORING
● ACTIVE

RECORDING
● ACTIVE

DATA
● UPDATING

Technical details belong in the diagnostics drawer.

The top-level interface should not make operators feel like they need to understand the AI architecture.

---

## 14. DIAGNOSTICS DRAWER

Keep the existing collapsible diagnostics concept.

This is where technical information belongs.

Example:

CAMERA DIAGNOSTICS

Resolution
1920 × 1080

Frame rate
29.8 FPS

Latency
38 ms

Packet loss
0.2%

Last frame
0.4 s ago

SYSTEM DIAGNOSTICS

Detection model
YOLOv5s

Tracking
SORT

Inference
24.8 ms

Device
GPU

Recording
Active

This information is useful for technicians/admins but should not dominate the operator interface.

---

## 15. OFFLINE / DEGRADED STATES

Keep these states.

They are operationally important.

OFFLINE:

CAMERA
OFFLINE

SIGNAL LOST

Do not display stale values as if they are live.

Clearly indicate:

Last known queue
18

Last update
14:32:18

Charts should show:

No signal

The global status should become OFFLINE.

DEGRADED:

CAMERA
DEGRADED

Image/data quality reduced

Keep the queue visible but clearly indicate that readings may be less reliable.

Do not falsely show a healthy green state when the monitoring pipeline is unavailable.

---

## 16. SIMULATION CONTROL

Remove the visible "Sim" control from the production-facing header.

The main dashboard should look like a real deployed application.

If simulation is needed for development/demo purposes, keep it in a developer/demo mode or hidden development control.

Do not make mock-data controls part of the normal operator experience.

---

## 17. COLOR SYSTEM

Use a restrained airport/security operations palette.

Background:
dark navy / blue-gray

Panels:
slightly lighter blue-gray

Borders:
subtle blue-gray

Primary text:
white

Secondary text:
cool gray

Normal:
green

Warning:
amber

Congestion:
orange/red

Offline:
muted red

Information:
blue

Avoid neon cyan and purple as dominant colors.

Do not use color purely for decoration.

Every status color should communicate meaning.

---

## 18. TYPOGRAPHY

Use:

Inter for UI and labels.

JetBrains Mono or another tabular font for dynamic numerical values.

Create a clear hierarchy:

Large:
queue status and people count

Medium:
KPIs

Small:
supporting metadata

Very small:
technical diagnostics

Do not make every number huge.

---

## 19. CAMERA VISUALIZATION

The camera feed is one of the most important visual elements.

Make it feel like an actual fixed airport CCTV camera:

- realistic perspective
- subtle timestamp
- camera ID
- restrained overlay
- no excessive sci-fi effects

YOLO bounding boxes should be thin and unobtrusive.

Tracking IDs should be small.

ROI should use a subtle boundary.

Entry/exit line should be visible but not dominant.

The operator should be able to visually understand the physical queue.

---

## 20. RESPONSIVENESS

Primary target:

1440 × 900

No unnecessary scrolling for the main desktop monitoring view.

Maintain the priority hierarchy at smaller sizes.

Do not simply shrink everything.

Collapse lower-priority technical details before shrinking the primary queue information.

---

## 21. MOCK DATA CONSISTENCY

Keep the existing single telemetry model.

All displayed values must remain logically consistent.

For example:

Queue = 18
Capacity = 30
Occupancy = 60%

Entry rate = 12/min
Exit rate = 8/min

Net change = +4/min

Queue trend = Increasing

Throughput = 8 × 60 = 480/hour

Waiting time must also remain internally consistent with the simulation model.

Do not create independent random numbers for individual widgets.

---

## 22. IMPORTANT SCOPE

This remains a CAMERA-ONLY prototype.

Do NOT introduce:

- MQTT
- cloud processing
- physical sensors
- sensor fusion
- passenger identity
- facial recognition
- facial embeddings
- external IoT systems

Do not imply these features already exist.

The dashboard can be visually prepared for future multi-camera expansion, but the current product is camera-only.

---

## 23. FINAL DESIGN TEST

Before finishing, evaluate the result using this question:

"If I did not know this system used AI, would I believe this is a real airport security operations monitoring dashboard?"

The answer should be YES.

The AI should feel like infrastructure underneath the product.

The dashboard should communicate:

"Here is the current condition of the security queue."

Not:

"Look at our AI technology."

Preserve the existing functionality and approved information architecture.

This is an evolution of the current design, not a complete redesign from zero.
