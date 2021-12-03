from pypdb.clients.search.search_client import perform_search_with_graph
from pypdb.clients.search.search_client import ReturnType
from pypdb.clients.search.search_client import QueryGroup, LogicalOperator
from pypdb.clients.search.operators import text_operators

#the header matching search operators
search_name_operator = text_operators.DefaultOperator(value="tau")


#enclude human only
is_human_operator = text_operators.ExactMatchOperator(
            value="Homo sapiens",
            attribute="rcsb_entity_source_organism.taxonomy_lineage.name")  #rcsb_entity_source_organism is part of the schema, taxonomy_lineage and name are its subparts in the json!

#operator for quality
under_4A_resolution_operator = text_operators.ComparisonOperator(
       value=3,
       attribute="rcsb_entry_info.resolution_combined",
       comparison_type=text_operators.ComparisonType.GREATER)

#combine operators into a query and define what you want returned

combined_query = QueryGroup(
    queries = [search_name_operator , is_human_operator, under_4A_resolution_operator],
    logical_operator = LogicalOperator.AND
)

return_type = ReturnType.ENTRY  #get the pdb entry id of the objects requested

#preform the search and do as you will with the results

results = perform_search_with_graph(
  query_object=combined_query,
  return_type=return_type)
print("\n", results[:10]) # first 10