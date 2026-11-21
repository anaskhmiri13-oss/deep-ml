def find_treasure(start_x: float) -> float:
  if start_x==3 : 
    l=1.0 
  else : 
     l= start_x
  r=3.0
  for i in range(100):
    mid = (l+r)/2
    gradient=4*mid**3-9*mid**2
    if gradient<0 : 
      l=mid 
    else : 
      r= mid 
  return (l+r)/2