ALE nextgen undated version
============================
For more details please consult our manuscript "Efficient Exploration of the Space of Reconciled Gene Trees"
(Syst Biol. 2013 Nov;62(6):901-12. doi: 10.1093/sysbio/syt054.)
and the original implementation of ALE at https://github.com/ssolo/ALE.


Install Python package requirements (Linux):
--------------------------------------------
```
python3 -m venv alengu_venv
source alengu_venv/bin/activate
pip install -r requirements.txt
```

Run the program:
----------------
```
python -m alengu.main examples/test_species_tree_1.newick examples/test_gene_tree_1a.newick
```

Programatical entry point:
--------------------------
```
main.run(loader, opt=True, native=False, undated_model=True)
```