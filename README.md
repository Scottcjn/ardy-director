# Director Service with Camera and Staging Control

## Introduction

This bounty adds camera and staging control to the Director service so the LLM can frame the shot and place the action.

## New Tools/Endpoints

### Set Camera Mode

- **Endpoint**: `/set_camera_mode`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "mode": "follow"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Camera mode set to: follow"
  }
  ```

### Place Waypoints

- **Endpoint**: `/place_waypoints`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "waypoints": [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0]
    ]
  }
  ```
- **Response**:
  ```json
  {
    "message": "Waypoints placed successfully",
    "root_path": "RootPath(waypoints=[[0, 0, 0], [1, 0, 0], [1, 1, 0]])"
  }
  ```

## Worked Example

### Stage a Path

1. Place waypoints for the character to follow.
   ```bash
   curl -X POST http://localhost:5000/place_waypoints -H "Content-Type: application/json" -d '{"waypoints": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}'
   ```

### Pick a Follow Camera

2. Set the camera mode to follow.
   ```bash
   curl -X POST http://localhost:5000/set_camera_mode -H "Content-Type: application/json" -d '{"mode": "follow"}'
   ```

### Generate

3. Generate the animation with the staged path and follow camera.
   ```bash
   curl -X POST http://localhost:5000/ardy_generate -H "Content-Type: application/json" -d '{"constraints": {"root_path": "RootPath(waypoints=[[0, 0, 0], [1, 0, 0], [1, 1, 0]])"}}'
   ```

## Conclusion

With these new tools and endpoints, you can now control the camera and stage the action in your animations. For more details, refer to the source code and the interactive demo scripts.