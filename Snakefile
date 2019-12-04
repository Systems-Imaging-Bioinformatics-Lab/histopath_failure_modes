"""Workflow for making predictions from TCGA slide images using pretrained DeepPATH models
"""

from pathlib import Path

input_dir = Path("work/input/")
intermeiate_dir = Path("work/intermediate/")
output_dir = Path("work/output/")
src_path = Path("DeepPATH/DeepPATH_code/").absolute()
ckpt = "run2a_3D_classifier/"

experiment_dataset = "phase1_exp"
N_EXPERIMENTS = 50
experiments = {
    "bubbles" : [100,20],
    "marker" : [15,80],
    "fold": [20],
    "sectioning": [15,30],
    "illumination" : [100,50,10]
}
exp_datasets = [f"exp_{exp_type}_{percent}"  for exp_type,percents in experiments.items() for percent in percents ]
SEED = 42
wildcard_constraints:
        dataset="\w+"

datasets = [ds.stem for ds in input_dir.iterdir() if ds.is_dir()] + exp_datasets

rule all:
    input: expand(str(output_dir / "{dataset}" / "auc"),dataset=datasets)

rule tile_images:
    input: directory(str(input_dir / "{dataset}"))
    output: directory(str(intermeiate_dir / "{dataset}" / "tiles/"))
    shell: 
        """
        python '{src_path}/00_preprocessing/0b_tileLoop_deepzoom4.py'  -s 512 -e 0 -j 32 -B 50 -M 20 -o {output} "{input}/*/*svs"  
        """
def find_metadata(wc):
    if wc.dataset.startswith("exp"):
        return [str(p.absolute()) for p in (input_dir / experiment_dataset).glob("metadata.*.json")]
    return [str(p.absolute()) for p in (input_dir / wc.dataset).glob("metadata.*.json")]
rule combine_jpeg_dir:
    input: 
        tiles=rules.tile_images.output,
        metadata= find_metadata
    output: directory(str(intermeiate_dir / "{dataset}" / "combine_jpg/"))
    shell:
        """
        mkdir -p {output}
        cd {output}
        echo pwd
        python '{src_path}/00_preprocessing/0d_SortTiles.py' --SourceFolder='../tiles' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=3  --PatientID=12 --nSplit 0 --JsonFile='{input.metadata}' --PercentTest=100 --PercentValid=0
        """
rule make_tf_record:
    input: str(intermeiate_dir /  "{dataset}"  /"combine_jpg/")
    output: directory(str(intermeiate_dir / "{dataset}" / "tf_records/"))
    shell:
        """
        mkdir -p {output}
        python '{src_path}/00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py' --directory='{input}'  --output_directory='{output}' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'
        """
rule predict:
    input: 
        data=rules.make_tf_record.output,
    output: str(output_dir / "{dataset}" / "out_filename_Stats.txt")
    shell:
        """
        mkdir -p $(dirname {output})
        python '{src_path}/02_testing/xClasses/nc_imagenet_eval.py' --checkpoint_dir=checkpoints/{ckpt} --eval_dir=$(dirname {output}) --data_dir={input.data}  --batch_size 300  --run_once --ImageSet_basename='test_' --ClassNumber 3 --mode='0_softmax'  --TVmode='test'
        """

rule agg_results:
    input: str(output_dir / "{dataset}" / "out_filename_Stats.txt")
    output:  directory(str(output_dir / "{dataset}" / "auc"))
    shell:
        """
        mkdir -p {output}
        python '{src_path}/03_postprocessing/0h_ROC_MultiOutput_BootStrap.py'  --file_stats {input}  --output_dir {output} --labels_names '{src_path}/example_TCGA_lung/labelref_r1.txt' --ref_stats '' 
        """

rule manipulate_tiles:
    input: 
        jpgs=str(intermeiate_dir / experiment_dataset / "tiles/")    
    output: 
        data=directory(str(intermeiate_dir / "exp_{exp_type}_{percent}" / "tiles/")),
    log:
        modified_tiles=str(output_dir/"manipulation_logs"/"{exp_type}_{percent}.txt"),
        bad_tiles=str(intermeiate_dir / "exp_{exp_type}_{percent}" / "log_bad_images.txt")
    shell:
        """
        rm -f {log.modified_tiles}
        rm -f {log.bad_tiles}
        cp -r {input.jpgs} {output.data}
        for slide in $(ls -d {output.data}/*/)
        do
            echo "$slide {wildcards.exp_type} {SEED}" > .seed.txt
            n_tiles=$(ls $slide/20.0/ | wc -l)
            n_changes=$(echo "$n_tiles*{wildcards.percent}/100" | bc)
            for image in $(ls $slide/20.0/ | sort -t " " -R --random-source=.seed.txt | head -n $n_changes)
            do
                image_file=$slide/20.0/$image
                set +e
                python image_manipulation/img_manip.py $image_file {wildcards.exp_type} $image_file 2>{log.bad_tiles} 
                exitcode=$?
                if [ $exitcode -gt 0 ]
                then
                    echo $image_file >> {log.bad_tiles}
                    echo "bad image: $image_file"
                else
                    echo $image_file >> {log.modified_tiles}
                fi
            done
        done
        """