from rcsbsearch import rcsb_attributes as attrs, TextQuery
#import nglview
# Create terminals for each query
q1 = TextQuery('"heat-shock transcription factor"')

q2 = attrs.rcsb_struct_symmetry.symbol == "C2"
q3 = attrs.rcsb_struct_symmetry.kind == "Global Symmetry"
q4 = attrs.rcsb_entry_info.polymer_entity_count_DNA >= 1
print(q2("assembly"))
# combined using bitwise operators (&, |, ~, etc)
query =  q1 & q2 & q3 & q4  # AND of all queries (something about q1 makes it break???)
print(vars(query))
#print(query("assembly")
# Call the query to execute it
for assemblyid in query("assembly"):
    print(assemblyid)