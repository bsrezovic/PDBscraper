import pypdb as pdb
import types


found_pdbs = pdb.Query(27499440, "PubmedIdQuery").search()
print(found_pdbs[:10])

q = pdb.Query("Dictyostelium", query_type="OrganismQuery")
print(q.search()[:10])

from pypdb.clients.search.search_client import perform_search
from pypdb.clients.search.search_client import SearchService, ReturnType
from pypdb.clients.search.operators import text_operators

search_operator = text_operators.DefaultOperator(value="ribosome")
return_type = ReturnType.ENTRY

results = perform_search(search_operator, return_type)

print(results[:10])