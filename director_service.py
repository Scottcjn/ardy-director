from flask import Flask, request, jsonify
import ardy.constraints as constraints

app = Flask(__name__)

# Existing endpoints
@app.route('/ardy_generate', methods=['POST'])
def ardy_generate():
    # Existing implementation
    pass

@app.route('/ardy_choreograph', methods=['POST'])
def ardy_choreograph():
    # Existing implementation
    pass

# New endpoints for camera and staging control
@app.route('/set_camera_mode', methods=['POST'])
def set_camera_mode():
    data = request.json
    mode = data.get('mode')
    if mode not in ['follow', 'orbit', 'fixed', 'over-the-shoulder']:
        return jsonify({'error': 'Invalid camera mode'}), 400
    # Set the camera mode (this is a placeholder for actual implementation)
    print(f"Camera mode set to: {mode}")
    return jsonify({'message': f'Camera mode set to: {mode}')

@app.route('/place_waypoints', methods=['POST'])
def place_waypoints():
    data = request.json
    waypoints = data.get('waypoints')
    if not waypoints:
        return jsonify({'error': 'Waypoints are required'}), 400
    # Place waypoints and feed them into ARDY's kinematic constraints
    root_path = constraints.RootPath(waypoints)
    print(f"Waypoints placed: {waypoints}")
    return jsonify({'message': 'Waypoints placed successfully', 'root_path': root_path})

if __name__ == '__main__':
    app.run(debug=True)