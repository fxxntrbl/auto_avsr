import torch
import torchaudio
from pytorch_lightning import LightningModule

from datamodule.transforms import TextTransform
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(
                self.cfg.pretrained_model_path,
                map_location=lambda storage, loc: storage,
            )
            if self.cfg.transfer_frontend:
                tmp_ckpt = {
                    k: v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("trunk.") or k.startswith("frontend3D.")
                }
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {
                    k.replace("encoder.", ""): v
                    for k, v in ckpt.items()
                    if k.startswith("encoder.")
                }
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt)

    def forward(self, video, audio):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)
        audiovisual_feat = self.model.fusion(
            torch.cat((video_feat, audio_feat), dim=-1)
        )

        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace(
            "<eos>", ""
        )
        return predicted


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None,
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
