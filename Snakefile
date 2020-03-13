"""Workflow for making predictions from TCGA slide images using pretrained DeepPATH models
"""

from pathlib import Path
work_dir = Path("work/")
input_dir = work_dir / "input"
intermeiate_dir = work_dir / "intermediate"
output_dir = work_dir / "output"
src_path = Path("DeepPATH/DeepPATH_code/").absolute()
ckpt = "run2a_3D_classifier/"

experiment_dataset = "all"
experiments = {
   "marker" : [50,90]
}
exp_datasets = [f"exp_{exp_type}_{percent}"  for exp_type,percents in experiments.items() for percent in percents ]
SEED = 42
wildcard_constraints:
        dataset="\w+"
input_data = [ds.stem for ds in input_dir.iterdir() if ds.is_dir()]
datasets =  input_data + exp_datasets
manifests  = {ds.stem:list(ds.glob("*manifest*"))[0] for ds in input_dir.iterdir() if ds.is_dir()}

rule all:
    input: expand(str(output_dir / "{dataset}" / "auc_summary.txt"),dataset=datasets)
rule download:
    input: expand(str(input_dir / "{dataset}" / "slides_downloaded.done"),dataset=input_data)
def find_manfiest(wc):
    if wc.dataset.startswith("exp"):
        return [str(p.absolute()) for p in (input_dir / experiment_dataset).glob("gdc_manifest*")]
    return [str(p.absolute()) for p in (input_dir / wc.dataset).glob("gdc_manifest*")]
rule download_slides:
    input:  find_manfiest
    output: touch(str(input_dir / "{dataset}" / "slides_downloaded.done"))
    threads: 24
    shell:
        """
        gdc-client download -m {input} -d $(dirname {output}) -n {threads}
        """
rule tile_images:
    input: d=directory(str(input_dir / "{dataset}")), flag=str(input_dir / "{dataset}" / "slides_downloaded.done")
    output: directory(str(intermeiate_dir / "{dataset}" / "tiles/"))
    shell: 
        """
        python '{src_path}/00_preprocessing/0b_tileLoop_deepzoom4.py'  -s 512 -e 0 -j 32 -B 50 -M 20 -o {output} "{input.d}/*/*svs"  
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
    log: "logs/tensor_flow_predict_{dataset}.txt"
    resources:
        gpu=1
    shell:
        """
        mkdir -p $(dirname {output})
        python '{src_path}/02_testing/xClasses/nc_imagenet_eval.py' --checkpoint_dir=checkpoints/{ckpt} --eval_dir=$(dirname {output}) --data_dir={input.data}  --batch_size 300  --run_once --ImageSet_basename='test_' --ClassNumber 3 --mode='0_softmax'  --TVmode='test' 2> {log}
        """

rule agg_results:
    input: str(output_dir / "{dataset}" / "out_filename_Stats.txt")
    output:  str(output_dir / "{dataset}" / "auc"/ "out2_perSlideStats.txt")
    shell:
        """
        mkdir -p $(dirname {output})
        python '{src_path}/03_postprocessing/0h_ROC_MultiOutput_BootStrap.py'  --file_stats {input}  --output_dir $(dirname {output}) --labels_names '{src_path}/example_TCGA_lung/labelref_r1.txt' --ref_stats '' 
        """

rule summarise_auc:
    input: str(output_dir / "{dataset}" / "auc"/ "out2_perSlideStats.txt")
    output: str(output_dir / "{dataset}" / "auc_summary.txt")
    shell:
        """
        cd $(dirname {input})
        OUTPUT_DIR=$(dirname {input})
        rm -f {output}
        ls -tr out1_roc_data_AvPb_c1*  | sed -e 's/k\/out1_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        ls -tr out1_roc_data_AvPb_c2*  | sed -e 's/k\/out1_roc_data_AvPb_c2/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        ls -tr out1_roc_data_AvPb_c3*  | sed -e 's/k\/out1_roc_data_AvPb_c3/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        ls -tr out2_roc_data_AvPb_c1*  | sed -e 's/k\/out2_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        ls -tr out2_roc_data_AvPb_c2*  | sed -e 's/k\/out2_roc_data_AvPb_c2/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        ls -tr out2_roc_data_AvPb_c3*  | sed -e 's/k\/out2_roc_data_AvPb_c3/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//' >> ../auc_summary.txt
        """

rule manipulate_tiles:
    input: 
        jpgs=str(intermeiate_dir / experiment_dataset / "tiles/")    
    output: 
        data=directory(str(intermeiate_dir / "exp_{exp_type}_{percent}" / "tiles/")),
    log:
        modified_tiles=str(intermeiate_dir / "exp_{exp_type}_{percent}" / "log_modified_tiles.txt"),
        bad_tiles=str(intermeiate_dir / "exp_{exp_type}_{percent}" / "log_bad_images.txt")
    threads: 16
    shell:
        """
        rm -f {log.modified_tiles}
        rm -f {log.bad_tiles}
        cp -r {input.jpgs} {output.data}
        rm -f .tile_list
        for slide in $(ls -d {output.data}/*/)
        do
            echo "$slide {wildcards.percent} {SEED}" > .seed.txt
            n_tiles=$(ls $slide/20.0/ | wc -l)
            n_changes=$(echo "$n_tiles*{wildcards.percent}/100" | bc)
            for image in $(ls $slide/20.0/ | sort -t " " -R --random-source=.seed.txt | head -n $n_changes)
            do
                image_file=$slide/20.0/$image
                echo $image_file >> .tile_list
            done
        done
        manip_image() {{
            set +e
            python image_manipulation/img_manip.py $1 {wildcards.exp_type} $1 2>>{log.bad_tiles} 
            exitcode=$?
            if [ $exitcode -gt 0 ]
            then
                echo $1 >> {log.bad_tiles}
                echo "bad image: $1"
            else
                echo $1 >> {log.modified_tiles}
            fi
        }}
        export -f manip_image
        parallel -a .tile_list -j {threads} manip_image 
        """