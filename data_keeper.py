""" Time Series Data Handling """
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import table, column, select, update, insert
from sqlalchemy.sql import and_, or_, not_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import InvalidRequestError, OperationalError, SQLAlchemyError
from sqlalchemy.engine.reflection import Inspector
from config import SQLALCHEMY_DATABASE_URI
from datetime import datetime
import traceback
import parse
import functools # py3 does not include reduce()
from itertools import compress

class data_keeper():
    # SQL details and objects
    dbName = 'met'

    # To allow for a standardized metData table the following sensor types are supported
    sensor_types_supported = ['wspd', 'wdir', 'temp', 'pres']
    sensor_stats_supported = ['min', 'max', 'std', 'avg']
    sensor_heights_supported = [1, 2, 3, 4]
    sensor_channels_supported = ['a', 'b', 'c']

    def __init__(self, verbose=True, recreate_if_exists=False):
        # Create a connection to the server & if the tables don't exist, create
        try:
            # parameters for connecting to mysql DB -- move elsewhere for security
            user, pwd, host, port, db = 'saurav', 'saurav', 'localhost', ' ', 'met'

            # Create engine to check if DB exists. Otherwise create the DB
            # this engine queries for list of DBs
            mysql_engine = create_engine('mysql+mysqldb://{0}:{1}@{2}'.format(user, pwd, host))
            existing_dbs = [d[0] for d in mysql_engine.execute("SHOW DATABASES;")]

            # Create database if not exists
            if db not in existing_dbs:
                mysql_engine.execute("CREATE DATABASE {0}".format(db))
                print("No existing database. Created database {0}".format(db))
                
            # Go ahead and use this engine
            self.db_engine = create_engine('mysql+mysqldb://{0}:{1}@{2}/{3}'.format(user, pwd, host, db))                
            self.metadata = MetaData(bind=self.db_engine)
            
            # Init db if necessary (i.e. no tables exist). 
            self.init_db(recreate_if_exists=recreate_if_exists, verbose=verbose)

            self.Session = sessionmaker(bind=self.db_engine)
            self.Base = automap_base(metadata=self.metadata)
            self.Base.prepare(self.db_engine, reflect=True)
            self.metadata.reflect(self.db_engine, schema=self.dbName)

            # Keep a reflected set of tables to be used 
            self.project_tbl = Table('Project', self.metadata, autoload=True, autoload_with=self.db_engine)
            self.tower_tbl   = Table('Tower',   self.metadata, autoload=True, autoload_with=self.db_engine)
            self.metData_tbl = Table('MetData', self.metadata, autoload=True, autoload_with=self.db_engine)
            self.metFlags_tbl= Table('MetFlags',self.metadata, autoload=True, autoload_with=self.db_engine)
            self.sensor_tbl  = Table('Sensor',  self.metadata, autoload=True, autoload_with=self.db_engine)
            self.sensorType_tbl  = Table('SensorType',  self.metadata, autoload=True, autoload_with=self.db_engine)            
            
            if verbose == True:
                print('Reflection complete')
        except InvalidRequestError as err:
            print(err.message)
            print('Error reflecting database.\n')
            traceback.print_exc()

    def init_db(self, recreate_if_exists=False, verbose=True):
        """ Create tables if they don't exist. 
        if 'recreate_if_exists' is set to True then drop existing & recreate """

        if recreate_if_exists == True:
            self.metadata.drop_all()

        # if tables exist, skip. otherwise create the metModel tables.
        # assumes existence of `Project` as the existence of all other tables.
        if (self.db_engine.dialect.has_table(self.db_engine.connect(), 'Project') == True):
            if verbose == True:
                print('DB tables already exist. Rerun with `recreate_if_exists=True` if you want to delete existing tables.')
        else:              
            # Open and read the file as a single buffer
            f = open('metModel.sql', 'r')
            sqlFile = f.read()
            f.close()

            # all SQL commands (split on ';')
            sqlCommands = sqlFile.split(';')

            # Execute every command from the input file
            for cmd in sqlCommands:
                try:
                    self.db_engine.execute(cmd)
                except:
                    print('Command skipped: %s' %cmd)
                    raise OperationalError()    # If an error is raise the execution will not continue

            print('DB tables created. You are all set to go!')
        
    # --------------------------------------------------------------------------#
    # Function for adding data on meta data on project, tower and sensors
    # --------------------------------------------------------------------------#
    def add_project(self, name, lat=None, lon=None, start_date=None):
        """ Create a new project """
        # Optionally do some checks to make sure the name is unique enough -- TO FIX
        s = self.Session()
        try:
            ins = self.Base.classes.Project(name=name, latitude=lat, longitude=lon,
                                            start_date=start_date.strftime('%Y-%m-%d'))            
            q = s.add(ins)
            s.commit()            
            return int(ins.id)
        except:
            s.rollback() # this rolls back the transaction unconditionally
            raise                    
        s.close()        
        
    def add_tower_to_project(self, name, parent_project, lat=None, lon=None,tower_type=None, start_date=None, end_date=None):
        """ Create a tower that is associated with the projects """
        s = self.Session()
        try:
            ins = self.Base.classes.Tower(name=name, project_id=parent_project,
                                          latitude=lat, longitude=lon, type=tower_type,
                                          start_date=start_date, end_date=end_date)
            q = s.add(ins)
            s.commit()            
            return int(ins.id)
        except:
            s.rollback()    # this rolls back the transaction unconditionally
            raise                    
        s.close()

    def add_sensor_to_tower(self, colName, parent_tower, deployed_height=None):
        """ Add the various sensors for this tower has """

        s = self.Session()
        try:
            parm, stat, sensorID = colName.split('_')
            
            # get id from name for sensorType -- e.g., 'wspd_avg'
            name = '_'.join([parm, stat])
            r = s.query(self.Base.classes.SensorType).\
                filter(self.sensorType_tbl.c.name == name).one()

            i = self.Base.classes.Sensor(name=colName, tower_id=parent_tower,
                                         sensorType_id=r.id,
                                         deployed_height=deployed_height)
            q = s.add(i)
            s.commit()
            return int(i.id)
        except:
            s.rollback() # this rolls back the transaction unconditionally
            raise
        s.close()                

    # --------------------------------------------------------------------------#
    # Function for ingesting time series
    # --------------------------------------------------------------------------#
    def add_metData(self, met_df, tower_name_id, col_format_str, col_format_content):
        """ Loads the data provided in a pandas data frame into the metData table.

            met_df: A pandas dataframe that is expected to have a timestamp column named 'ts' 
                    followed by sensor data columns named using a specific format string ('type_stat_height_channel') 

                    'type' can be ['wspd', 'wdir', 'temp', 'pres']. 
                    'stat' can be ['min', 'max', 'std', 'avg'].
                    'height' can be [1, 2, 3, 4] where 1 is the highest sensor and 4 the lowest.
                    'channel': can be ['a', 'b', 'c'] corresponding to primary, secondary and tertiary sensor at same height.

                    Valid examples of met_df columns names are 'ws_avg_1a' or 'wdir_min_2b' etc.

            tower_name_id: Either an integer corresponding to the tower 'id' or a 'name' string that retrieves a unique tower id
        """
        s = self.Session()
        
        # Validate and retrieve tower info
        found_tower = self.retrieve_tower_info(tower_name_id)
        tower_id = found_tower['id']

        # Check the columns of data frame before inserting.
        # Delete columns not found in database
        try:
            met_df = met_df.copy()  # Make a copy so that the original df remains unchanged
            metData_cols = [c.key for c in self.metData_tbl.c]
            met_df.columns = self.translate_met_col_names(met_df.columns.tolist(),
                                                          col_format_str, col_format_content)

            # Find columns that are in met_df but not in metData_columns
            unrecognized_cols = set(met_df.columns) - set(metData_cols)
            if unrecognized_cols:
                met_df.drop(unrecognized_cols, axis=1, inplace=True)

            # Set 'id' to tower_id
            met_df['tower_id'] = tower_id

            # add each of the sensors to the `Sensor` table
            cols = [ x for x in met_df.columns[met_df.columns.str.contains('wspd|wdir|temp')] ]
            for col in cols:
                height = 0 # retrieve height eventually
                self.add_sensor_to_tower(col, tower_id, deployed_height=height)
            
            # Delete any existing data -- there are no cases where we need to append???
            # i = s.query(self.Base.classes.MetData).filter(self.metData_tbl.c.tower_id == tower_id).\
            #     filter(self.metData_tbl.c.timestamp.between(met_df.timestamp.min().strftime('%Y-%m-%d %H:%M:00'),
            #                                                 met_df.timestamp.max().strftime('%Y-%m-%d %H:%M:59'))).all()
            # s.delete(i)

            # Add data to SQL DB
            met_df.to_sql(self.metData_tbl.name, self.db_engine, if_exists='append',
                          chunksize=10000, index=False)

            s.commit()
        except SQLAlchemyError as err:
            raise

        s.close()
        
            
    def set_dataFlags(self, tower_id, ts, mask_name, mask):
        """ Save the QC flags from the raw data into the flags DB."""
        # Use some logic from the tower / sensor combination to figure which column this 
        # Currently, setting flags one by one. SUPER SLOW in I/O. FIX.

        s = self.Session()
        try:
            # retrieve all data for that tower
            q = "SELECT * FROM %s WHERE tower_id = %s ORDER BY timestamp" \
                %(self.metFlags_tbl.name, tower_id)

            df = pd.read_sql(q, self.db_engine, index_col='timestamp')#, parse_dates=('timestamp'))

            df[mask_name] = mask
            df.to_sql(self.metFlags_tbl.name, self.db_engine, if_exists='replace',
                      chunksize=10000)
            
            # x = df[mask_name] == True # testing, testing
            # print(np.shape(df[x]), np.shape(df[~x]))
            s.commit()
        except:
            s.rollback()
            raise
            print('ERROR: Could not write to met.DataFlags. sensor_name = %s' %mask_name)
        s.close()

    def get_sensor_column_name(self, sensor_id):
        """ Map the sensor type to a column name in the sensor data table """

        sensor_type = get_sensor_type_from_id(sensor_id)
        if sensor_type == 'wspd':   # This is just an example
            return 'wspd_1_min';
        else:
            return 'wspd_1_mean'

    def set_calculated_value(self, tower, name, timestamp, values):
        """ Set a value that is calculated of the time series e.g. Ratio of wind speeds
            tower: Either tower id or name that the sensor is located on e.g. Tower 4 or 'Met6'
            name: Name of the calculated value. e.g. top_wspd_ratio , that can be used in the recall of this value
        """
        pass
    
        # Easiest to do this with pandas data frame
        df = pd.DataFrame({'tower':tower, 'name':name, 'ts': timestamp, 'values': values})
        df.to_sql('calcualted_values_table', self.db_engine)

    # ---------------------------------------------------------------------------------#
    # Function for query time series data
    # ---------------------------------------------------------------------------------#
    def get_timeseries(self, tower_name_id, time_range='all', data_columns='all'):
        """ Return a time series from the SQL DB as a pandas df. 
            Inputs:
            tower_id_name: Either tower id or name that the sensor is located on 
                   e.g. 1, 'Tower 4',  or 'Met6'
            time_range: two element array giving the [min, max] of time range
            data_columns: A list containing , the following tuples (sensor type, stat, height, channel)
                    e.g. [ (wspd,avg,1,a), (wspd,std,1,b), (wdir,avg,2,a)] will return data for 
                        wind speed for top two anemometers and wind direction from the middle wind vane.
                    See, supported options on top
        """
        # Validate and retrieve tower info
        found_tower = self.retrieve_tower_info(tower_name_id)
        tower_id = found_tower['id']

        s = self.Session() # instantiate a session
        try:
            # Prepare query
            if isinstance(data_columns, str) and (data_columns.lower() == 'all'):
                q = "SELECT *"
            elif isinstance(data_columns, str) and (data_columns.lower() == 'timestamp'):
                q = "SELECT tower_id, timestamp"
            else:
                # col_names = [self.compose_met_col_name(x[0],x[1],x[2],x[3]) for x in data_columns]
                q =  "SELECT tower_id, timestamp, " + ", ".join(data_columns) 

            q += " FROM %s " %self.metData_tbl.name
            if time_range != 'all':
                q += "WHERE timestamp between '%s' and '%s' and tower_id = '%s' " \
                     %(time_range[0].strftime('%Y-%m-%d %H:%M:00'),
                       time_range[1].strftime('%Y-%m-%d %H:%M:00'),
                       tower_id)
            else:
                q += "WHERE tower_id = '%s' " %tower_id
            q += "ORDER BY timestamp"
                         
            print(q)

            # parse_dates needs to be a list or dict! http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql.html
            df = pd.read_sql(q, self.db_engine, index_col='timestamp')#, parse_dates=('timestamp'))

            return df
        except SQLAlchemyError as err:
            raise

        s.close()
        
    def get_masks(self, tower_name_id, mask_names='all', time_range='all'):
        """ Get values for a mask that was saved earlier """

        # Validate and retrieve tower info
        found_tower = self.retrieve_tower_info(tower_name_id)
        tower_id = found_tower['id']

        s = self.Session() # instantiate a session
        try:
            if isinstance(mask_names, str) and (mask_names.lower() == 'all'):
                q = "SELECT *"
            else:                
                q = "SELECT tower_id, timestamp, " + ', '.join(mask_names)

            q += " FROM %s " %self.metFlags_tbl.name
            if time_range != 'all':
                q += " WHERE timestamp between '%s' and '%s' and tower_id = '%s' " \
                     %(time_range[0].strftime('%Y-%m-%d %H:%M:00'),
                       time_range[1].strftime('%Y-%m-%d %H:%M:00'))
            q += "WHERE tower_id = '%s' ORDER BY timestamp" %tower_id

            print(q)
            df = pd.read_sql(q, self.db_engine, index_col='timestamp')#, parse_dates=('timestamp'))

            return df
        except SQLAlchemyError as err:
            raise
        s.close()

    def get_calculated_value(self, tower, name, timestamp):
        """ Get a calculated value that was stored earlier
            tower: Either tower id or name that the sensor is located on e.g. Tower 4 or 'Met6'
            name: Name of the calculated value. e.g. top_wspd_ratio , that can be used in the recall of this value
        """

        # Easiest to do this with pandas data frame
        pass

    # ---------------------------------------------------------------------------------#
    # Function for retrieving metadata
    # ---------------------------------------------------------------------------------#
    def retrieve_project_info(self, project_name_id):
        """ Retrieve info on the project this 'name' (if str) or 'id' (if an int) is provided.
            Return project 'id', 'name' .... and all other columns from the project 
        """

        s = self.Session()
        
        if isinstance(project_name_id, str):
            r = s.query(self.Base.classes.Project).filter(self.project_tbl.c.name == project_name_id).one()
        elif isinstance(project_name_id, int):
            r = s.query(self.Base.classes.Project).filter(self.project_tbl.c.id == project_name_id).one()
        else:
            raise ValueError('Input for project_name_id should be a string or an integer')
        
        try:
            return row_to_dict(r)
        except SQLAlchemyError as err:
            err.message = 'Error retreiving project details.' + err.message
            raise

        s.close()
        
    def retrieve_tower_info(self, tower_name_id):
        """ Retrieve info on the tower this 'name' (if str) or 'id' (if an int) is provided.
            Return tower 'id', 'name' .... and all other columns from the tower 
        """
        s = self.Session()

        if isinstance(tower_name_id, str):
            r = s.query(self.Base.classes.Tower).filter(self.tower_tbl.c.name == tower_name_id).one()
        elif isinstance(tower_name_id, int):
            r = s.query(self.Base.classes.Tower).filter(self.tower_tbl.c.id == tower_name_id).one()
        else:
            raise ValueError('Input for tower_name_id should be a string or an integer')
        
        try:
            return row_to_dict(r)
        except SQLAlchemyError as err:
            err.message = 'Error retreiving tower details.' + err.message
            raise

        s.close()
        
    # ---------------------------------------------------------------------------------#
    # Utility functions
    # ---------------------------------------------------------------------------------#
    def translate_met_col_names(self, cols, col_format_str, col_format_content):
        """ Attempts to make the myriad of naming conventions found in various met data and attempts to match it to 
                the internal format of 'type_stat_height-channel' e.g. 'ws_avg_1-a'
            col_format_str: Is a format recognizable by the parse library (https://pypi.python.org/pypi/parse/1.6.6)
            col_format_content: Specifies the order of sensor type, stat, height, channel & units

            For instance if the raw column name is 'ANEM_M/S_41M_CH3SD',
                column_format_str is '{:D}_{:D}_{:d}M_CH{:d}{:D}' and  column_format_content is ['type','unit','height','channel','stat']
        """
        raw_cols = list(cols)       # Keep old list around
        sensor_types_supported = ['wspd', 'wdir', 'temp', 'press']
        sensor_stats_supported = ['min', 'max', 'std', 'avg']
        sensor_heights_supported = ['1', '2', '3', '4']
        sensor_channels_supported = ['a', 'b', 'c']
        
        # Name translation dictionary
        key_word_translation = {'windspeed':'wspd', 'ws':'wspd', 'anem':'wspd', 'anemometer':'wspd',   # Wind Speed keywords
                                'winddirection':'wdir', 'direction':'wdir',   # Wind Direction keywords
                                'temperature': 'temp', 'pressure':'press',    # Temperature, pressure keywords
                                'minimum':'min', 'maximum': 'max',
                                'mean':'avg', 'average':'avg',
                                'stdev':'std', 'stddev':'std', 'sd':'std','standarddeviation':'std'}
        # Search and replace keywords in the column names
        cols = [functools.reduce(lambda x,y: x.replace(y,key_word_translation[y]), key_word_translation, x.lower()) for x in cols]  

        # Extract sensor type, stat, height and channels
        measurement = []; stat = []; height = []; channel = []
        
        #import pdb; pdb.set_trace()
        for col in cols:
            parsed = parse.parse(col_format_str, col)
            measurement.append(parsed[col_format_content.index('type')] if parsed else '')
            stat.append(parsed[col_format_content.index('stat')] if parsed else '')
            height.append(parsed[col_format_content.index('height')] if parsed else '')
            channel.append(parsed[col_format_content.index('channel')] if parsed else '')

        # Make a data frame 
        dfm = pd.DataFrame()
        dfm['type'] = [find_matching_list_element(sensor_types_supported, x) for x in measurement]
        dfm['stat'] = [find_matching_list_element(sensor_stats_supported, x) for x in stat]
        dfm['height'] = height      # This needs to be converted to ranking
        dfm['channel'] = channel    # This needs to be converted to ranking
        dfm['old_name'] = raw_cols 

        # List all the sensor heights by sensor type and get their ranking in descending order with 1 being the highest
        def rank_height(sensor_type, height):
            all_heights = dfm[dfm.type==sensor_type].height.unique().tolist()
            all_heights.sort()
            return sensor_heights_supported[all_heights.index(height)]

        dfm['height_rank'] = dfm.apply(lambda row: rank_height(row['type'], row['height']), axis=1)
        
        # Rank by channel number
        def rank_channel(sensor_type, stat, height_rank, channel):
            all_channels = dfm[(dfm.type==sensor_type) &
                               (dfm.height_rank==height_rank) &
                               (dfm.stat == stat)].channel.unique().tolist()
            all_channels.sort()
            return sensor_channels_supported[all_channels.index(channel)]

        dfm['channel'] = dfm.apply(lambda r: rank_channel(r['type'], r['stat'],
                                                          r['height_rank'], r['channel']), axis=1)
        
        dfm['new_name'] = dfm.apply(lambda r: self.compose_met_col_name(r['type'],
                                                                        r['stat'],
                                                                        r['height_rank'],
                                                                        r['channel'])
                                    if r['type'] else r['old_name'], axis=1)
        return dfm.new_name.values.tolist()

    def compose_met_col_name(self, sensor_type, sensor_stat, height_rank, channel):
        """ The function composes a standard column name by combining 
            sensor type (e.g. 'ws'), sensor stat (e.g. min, max)
            height rank (e.g. 1, 2,..), channel (e.g. a, b, ...)
        """
        return '%s_%s_%s%s' % (sensor_type, sensor_stat, height_rank, channel)



def list_to_dict(header,data):
    """ Takes a header = ['h1','h2',...,'h3n'] and data = [ [y1,y2,...,yn],[z1,z2,...,zn] ,...]
        and converts it to {['h1':y1,'h2':'y2',...,'hn':yn],['h1':z1,'h2':'z2',...,'hn':zn] }
        This dict structure when jsonified works better with several underscorejs functions in javascript
    """
    return [dict(zip(header,x)) for x in data]

def row_to_dict(row):
    """ Returns a dict from a python object with a single row."""

    d = {}
    for column in row.__table__.columns:
        d[column.name] = str(getattr(row, column.name))
    return d
    #row_to_dict = lambda r: {c.name: str(getattr(r, c.name)) for c in r.__table__.columns}
    
def find_matching_list_element(list_to_match, value):
    """ Return the elements of the list that are contained in value. 
            e.g. find_matching_list_element(['apple','orange','pear'],'an apple') returns 'apple', 
                since apple was the only list element in the value
    """
    return_val =  list(compress(list_to_match, [x in value for x in list_to_match]))
    return return_val[0] if return_val else value


if __name__ == '__main__':
    from sqlalchemy import inspect
    engine = create_engine(SQLALCHEMY_DATABASE_URI,
                           encoding='latin1', echo=False)
    inspector = inspect(engine)
    
    for table_name in inspector.get_table_names():
        print("Table: %s" % table_name)
        if table_name == 'sensor' or table_name == 'project':
            for column in inspector.get_columns(table_name):
                print("\tColumn: %s" % column['name'])   


# Example usage of data keeper

# Lets suppose we have read the Excel file and ready to move it into our database

# 0. Create a data keeper instance to handle data storage
#       dk = data_keeper()      
# 1. Create project 
#       project_id = dk.add_project('TestProject', 34.6, 124.3, datetime(2013,1,1))
# 2. Add tower to this project
#       met1_id = dk.add_tower('Met1', project_id, 34.5, 124.1, datetime(2013,4,4))
# 3. Add sensor to this tower
#       sensor_1 = dk.add_sensor_to_tower(met1_id, 'top_primary_anemo', 'wspd_mean')
#       sensor_2 = dk.add_sensor_to_tower(met1_id, 'top_secondary_anemo', 'wspd_mean')
#       sensor_3 = dk.add_sensor_to_tower(met1_id, 'top_wind_vane', 'wdir_mean')
#       ...
# 4. Add data from excel
#       dk.set_time_series(project_id, met_df, [sensor_1, sensor_2,sensor_3,...])
