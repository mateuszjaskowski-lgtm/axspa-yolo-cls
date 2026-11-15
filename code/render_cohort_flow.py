from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CAPTION = ('Figure S1. Cohort selection. The reconstructed totals before exclusions were 716 patients in Łódź and 163 in Końskie. '
           'M.P. reviewed the radiographs for DISH, fractures and evidence of previous spinal surgery, excluding 33 and 8 cases, respectively. '
           'A further 9 and 5 cases were excluded for artifacts or poor image quality; these were additional, non-overlapping exclusions. '
           'The final cohorts contained 674 patients in Łódź (579 training, 95 internal validation) and 150 in Końskie (external evaluation), '
           'with one retained image per patient. Reasons are grouped without implying the order of eligibility and quality checks. '
           'Severe degenerative changes were not an exclusion criterion.')

def render_cohort_flow(output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.4,5.7));fig.subplots_adjust(left=.02,right=.98,bottom=.01,top=.99)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.965,'879 patients with chronic back pain\nReferred for spinal radiography by a rheumatologist',ha='center',va='top',fontsize=11.2,color='#172C43',linespacing=1.4)
    def box(x,y,w,h,text,fill='#F2F5F8',size=10.7,bold=False):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',linewidth=1.05,edgecolor='#425A70',facecolor=fill))
        ax.text(x,y,text,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',linespacing=1.45,color='#172C43')
    def arrow(x,top,bottom):
        ax.annotate('',xy=(x,bottom),xytext=(x,top),arrowprops={'arrowstyle':'-|>','lw':1.2,'color':'#425A70','shrinkA':0,'shrinkB':0})
    for x,center,n,review,retained,last in [
      (.25,'Łódź',716,'Eligibility and quality review\n42 excluded\nDISH: 8; fractures: 13\nPrevious spinal surgery: 12\nArtifacts / poor quality: 9',674,'Training: 579\nInternal validation: 95'),
      (.75,'Końskie',163,'Eligibility and quality review\n13 excluded\nDISH: 2; fractures: 3\nPrevious spinal surgery: 3\nArtifacts / poor quality: 5',150,'External evaluation: 150')]:
        box(x,.805,.44,.13,f'{center}\nAssessed: {n}',bold=True)
        arrow(x,.731,.676)
        box(x,.505,.44,.32,review,fill='#FFFFFF',size=10.4)
        arrow(x,.337,.282)
        box(x,.21,.44,.125,f'Retained cohort: {retained}',bold=True)
        arrow(x,.1395,.101)
        box(x,.055,.44,.075,last,fill='#FFFFFF',size=10.2)
    for ext in ('png','pdf'):
        fig.savefig(output_dir/f'Figure_S1_cohort_flow.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    return output_dir/'Figure_S1_cohort_flow.png'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Render the author-corrected cohort flow without model execution.')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'reproduced_figures')
    args=parser.parse_args()
    render_cohort_flow(args.output)
