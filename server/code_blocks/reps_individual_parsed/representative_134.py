#deltaToBounds = []
#for v in deltaToDist:
#    topBound = numpy.percentile(v[1], 95)
#    bottomBound = numpy.percentile(v[1], 5)
#    deltaToBounds.append([v[0], (topBound, bottomBound)])

def computePercentilesRange(df):
  import numpy
  import pandas
  from pandas.api.types import is_numeric_dtype

  lowerPercentile = 5
  upperPercentile = 95
  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  data = {"Percentile"+str(lowerPercentile):[],"Percentile"+str(upperPercentile):[],"columnName":quantitativeColumns}
  for c in quantitativeColumns:
    data["Percentile"+str(lowerPercentile)].append(numpy.percentile(df[[c]],lowerPercentile))
    data["Percentile"+str(upperPercentile)].append(numpy.percentile(df[[c]],upperPercentile))
  return pandas.DataFrame(data)
  
