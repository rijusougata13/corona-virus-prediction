# corona-virus-prediction
this is a small project about corona virus death prediction
feel free to add your idea..
their is lack of style feel free to modify this.
-->
user can give a date in mm/dd/year format and get a expected value of death worldwide
there is also a graph descibing it

-->
i used this code to convert date to the usable format-->
       df['date'] = pd.to_datetime(df['date'])
       df['date']=df['date'].map(dt.datetime.toordinal)
but,    
i cant use the actual date format in the graph so please help me with it.

stay clean,stay safe from corona virus
