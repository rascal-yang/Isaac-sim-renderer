"""Regression: producer may finish while eight GPU frames are still in flight."""
import socket,threading,struct,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from state_stream import StateRecorder

def test_close_waits_for_pending_frame_acks():
 p,c=socket.socketpair();errors=[]
 def consume():
  try:
   end=c.recv(8);assert struct.unpack('!q',end)[0]==-1
   for frame in range(4,11):c.sendall(struct.pack('!q',frame))
   assert c.recv(1)==b''
  except BaseException as e:errors.append(e)
  finally:c.close()
 with tempfile.TemporaryDirectory() as d:
  r=StateRecorder.__new__(StateRecorder);r.socket=p;r.last_ack=3;r.capture=SimpleNamespace(control_index=11)
  r.contacts=r.net_contacts=r.physics=r.frames=r.cameras=SimpleNamespace(flush=lambda:None);r.out=Path(d);r.publish_wait_s=0.;r.count=44;r.write_s=0.;r.started=r.started_wall=0.;r.frame_times=[];r.queue_events=[]
  th=threading.Thread(target=consume);th.start();r.close();th.join(timeout=2)
  assert not th.is_alive() and not errors,errors
  assert r.last_ack==10 and json.loads((Path(d)/'state_recording_timing.json').read_text())['physical_steps']==44
if __name__=='__main__':test_close_waits_for_pending_frame_acks();print('stream close regression passed')
