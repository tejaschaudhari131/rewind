from pathlib import Path
import sys, asyncio, json
sys.path.insert(0,str(Path(__file__).resolve().parent/'audit-deps'))
import cv2, numpy as np
from recommended_primitives import pixels_to_height_plane_mm, top_plane_homography, collect_before_deadline

async def main():
    K=np.array([[1000.,0,640],[0,1000,400],[0,0,1]])
    R=cv2.Rodrigues(np.array([.12,-.09,.03]))[0] @ np.diag([1.,-1.,-1.])
    C=np.array([200.,125.,500.]); t=-R@C
    dist=np.array([.02,-.01,.001,-.002,0.])
    points=np.array([[110.,120.,4.],[260.,120.,4.],[180.,180.,4.]])
    pixels=cv2.projectPoints(points,cv2.Rodrigues(R)[0],t,K,dist)[0]
    recovered=pixels_to_height_plane_mm(pixels,K,dist,R,t,4.)
    error=float(np.max(np.abs(recovered-points[:,:2])))
    assert error < 1e-4
    H=top_plane_homography(K,R,t,4.)
    undistorted=cv2.undistortPoints(pixels,K,dist,P=K).reshape(-1,2)
    projected=cv2.perspectiveTransform(points[:,:2].reshape(-1,1,2),H).reshape(-1,2)
    assert np.max(np.abs(projected-undistorted)) < 1e-4
    loop=asyncio.get_running_loop()
    fast=await collect_before_deadline(asyncio.sleep(.001,result=42), started_at=loop.time(),deadline_seconds=.1)
    slow=await collect_before_deadline(asyncio.sleep(.2,result=42), started_at=loop.time(),deadline_seconds=.02)
    assert fast.status=='complete' and fast.values==(42,)
    assert slow.status=='unknown_timeout' and slow.values==()
    result={'synthetic_projection_max_abs_mm':error,'deadline_fast':fast.status,'deadline_slow':slow.status,'cadquery':'not executed; kernel unavailable'}
    (Path(__file__).resolve().parent/'recommendation-test-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

asyncio.run(main())
