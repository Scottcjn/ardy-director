import asyncio
import websockets
import json
import numpy as np
import viser
import trimesh

#... (existing imports and code)

async def receive_motion(websocket):
    async for message in websocket:
        data = json.loads(message)
        motion = data['motion']
        local_rot_mats = np.array(motion['local_rot_mats'])
        root_positions = np.array(motion['root_positions'])
        posed_joints = np.array(motion['posed_joints'])

        # Update the character with the new motion data
        for i in range(local_rot_mats.shape[0]):
            character.update_pose(local_rot_mats[i], root_positions[i], posed_joints[i])

async def main():
    uri = "ws://localhost:9600/ws"
    async with websockets.connect(uri) as websocket:
        await receive_motion(websocket)

# Start the WebSocket connection
asyncio.get_event_loop().run_until_complete(main())

# Start the viser viewer
viewer = viser.ViserServer()
viewer.start()