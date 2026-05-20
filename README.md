# CS298-cfDNA
Fragmentomics - Characterization and Detection of Cancer using cfDNA fragmentomes includes two machine learning pipelines. The first pipelines allow the use of global features to learn characteristics of cancer related fragment length patterns. The second pipeline calculates Shannon Entropy at exon 1 regions of each sample to identify associated genes 

Ensure conda is installed
conda init
	•	conda activate ngs #for all bam files 
	•	****conda install bioconda: :samtools - to install samtools (not required everytime)

________________________________________________________________________
To download hg19 reference genome (canonical_exon1_hg19_protein_coding): 

1.  # change into a working folder first
mkdir -p /scratch/home/gbhogal/CS298/anno && cd $_
# Get GENCODE v19 GTF (GRCh37/hg19). If wget URL fails, use your browser to download and scp it up.
wget -O gencode.v19.annotation.gtf.gz "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
gunzip -f gencode.v19.annotation.gtf.gz


