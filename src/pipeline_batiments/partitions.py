"""
Definition des partitions Dagster : une partition = une orthophoto.

A implementer, ex :

    from dagster import DynamicPartitionsDefinition

    orthophoto_partitions = DynamicPartitionsDefinition(name="orthophotos")

Permet de matérialiser/rejouer le pipeline independamment pour chaque
orthophoto de test (cf jeu de 2-3 orthos fournies par l'equipe).
"""
