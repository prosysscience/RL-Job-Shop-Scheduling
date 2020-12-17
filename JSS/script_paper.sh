INSTANCES=('/JSS/JSS/env/instances/ta41' '/JSS/JSS/env/instances/ta42' '/JSS/JSS/env/instances/ta43' '/JSS/JSS/env/instances/ta44' '/JSS/JSS/env/instances/ta45'  '/JSS/JSS/env/instances/ta46' '/JSS/JSS/env/instances/ta47' '/JSS/JSS/env/instances/ta48' '/JSS/JSS/env/instances/ta49' '/JSS/JSS/env/instances/ta50')
 
for inst in "${INSTANCES[@]}"
do
  ./paper_results.py --instance_path="$inst"
done