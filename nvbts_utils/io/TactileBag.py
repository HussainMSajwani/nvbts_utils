from pathlib import Path
import dask.array as da
import json 


try:
    import rosbag
    from rospy import Time
except ImportError:
    print('ROS not found')

import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import h5py

class TactileBag:
    def __init__(self, path, desc) -> None:
        self.path = Path(path).resolve()
        self.desc = desc

    @property
    def bag_file(self):
        bag_file_name = list(self.path.glob('*.bag'))

        if len(bag_file_name) > 1:
            raise Warning(f'More than one bag found. Choosing {bag_file_name[0]}')
        
        bag_file_name = bag_file_name[0]
        self.bag_file = rosbag.Bag(bag_file_name)     

    def parse(
        self,
        start_time=0,
        topics=[],
        camera='DAVIS346',
        **kwargs
        ):
        params = {
            'path': str(self.path),
        }

        with open(self.path / 'params.json', 'w') as f:
            json.dump(params, f, indent=4)

        
        ############## parse events
        events = []
        contact_status = []
        contact_status_ts = []
        contact_angle = []
        contact_angle = []

        for topic, msg, t in tqdm(
            self.bag_file.read_messages(topics=['/dvs/events']), 
            total=sum([self.bag_file.get_message_count(top) for top in ['/dvs/events']]),
            desc='parsing events',
            unit='msg'
        ):
            if topic == '/dvs/events':
                ev_array = []
                for e in msg.events:
                    event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
                    ev_array.append(event)
                events.append(da.array(ev_array))
                
        da_event = da.vstack(events)
        da_event.to_hdf5(self.path / 'events.h5', 'events')


        ########## parse other
        if len(topics) == 0:
            return
        else:
            for topic, msg, t in tqdm(
                self.bag_file.read_messages(topics=topics, start_time=Time(start_time)), 
                total=sum([self.bag_file.get_message_count(top) for top in topics]),
                desc='parsing other',
                unit='msg'
            ):
                if topic == '/dvs/events':
                    for e in msg.events:
                        event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
                        events.append(event)
                elif topic == '/contact_status':
                    contact_status.append(msg.data)
                    contact_status_ts.append(t.to_nsec())
                elif topic == '/contact_angle':
                    contact_angle.append([msg.x, msg.y, msg.z])

                    # Updated contact status according to no. of events

            #print(events)
        self.bag_file.close()
      
        contact_angle = np.array(contact_angle)
        if '/contact_status' in topics:
            contact_status = np.array(contact_status)
            contact_status_ts = np.array(contact_status_ts)
        
        if '/contact_angle' in topics:
            contact_angle = np.array(contact_angle)

            df = pd.DataFrame({ 
                'ts': contact_status_ts,
                'contact_status': contact_status, 
                'contact_angle_x': contact_angle[:, 0],
                'contact_angle_y': contact_angle[:, 1],
                'contact_angle_z': contact_angle[:, 2],
                }
            )   
            df.to_csv(self.path / 'parsed_bag.csv', index=False)



    def is_parsed(self):
        events_h5_exists = (self.path / 'events.h5').exists()
        params_exist = (self.path / 'params.json').exists()
        #csv_exists = (self.path / 'parsed_bag.csv').exists()
        return (events_h5_exists and params_exist)

    def parse_exception(self):
        raise Exception('Bag not parsed yet. Call parse before loading.')

    @property
    def params(self):
        if not self.is_parsed():
            self.parse_exception()

        with open(self.path / 'params.json', 'r') as f:
            out = json.load(f)

        return out
        
    @property
    def events(self):
        if not self.is_parsed():
            self.parse_exception()
        return da.array(h5py.File(self.path / 'events.h5', 'r')['events'])

    @property
    def parsed_bag(self):
        if not self.is_parsed():
            self.parse_exception()
        return pd.read_csv(self.path / 'parsed_bag.csv')
        
        
