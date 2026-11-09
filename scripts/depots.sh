#!/usr/bin/env bash
# État et pull des deux dépôts git du dossier (public : .git, interne : .git-interne).
#
#   scripts/depots.sh etat   fetch, puis fichiers modifiés et avance/retard sur origin/main
#   scripts/depots.sh pull   fetch, puis pull --ff-only de chaque dépôt en retard
#
# Ne force jamais rien : un dépôt qui a divergé, ou dont un fichier modifié localement est
# touché par le pull, est laissé tel quel et signalé (code de sortie 1). Pas de commit ni de
# push ici : ils demandent de choisir les fichiers et le message.

set -u
cd "$(dirname "$0")/.." || exit 1

mode="${1:-etat}"
case "$mode" in
    etat | pull) ;;
    *)
        echo "usage : scripts/depots.sh etat|pull" >&2
        exit 2
        ;;
esac

probleme=0

# Un fetch SSH vers GitHub reste parfois suspendu plusieurs minutes (4 et 9 novembre) : on
# abandonne après ~45 s de silence au lieu d'attendre.
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3}"

un_depot() {
    local nom="$1"
    shift
    local g=("$@")
    echo "== $nom"

    if [ "$nom" = "interne" ]; then
        # Pas de refspec de fetch dans le dépôt interne : sans celui-ci, origin/main n'existe pas.
        "${g[@]}" fetch -q origin main:refs/remotes/origin/main
    else
        "${g[@]}" fetch -q origin
    fi
    if [ $? -ne 0 ]; then
        echo "   FETCH IMPOSSIBLE (réseau ou accès SSH) : relancer, rien n'est fait pour ce dépôt"
        probleme=1
        return
    fi

    local branche
    branche=$("${g[@]}" rev-parse --abbrev-ref HEAD)
    if [ "$branche" != "main" ]; then
        echo "   branche courante : $branche (attendu : main), rien n'est fait"
        probleme=1
        return
    fi

    # Le dépôt interne voit tout l'arbre de travail : on n'y liste que les fichiers suivis.
    local modifies
    if [ "$nom" = "interne" ]; then
        modifies=$("${g[@]}" status --short -uno)
    else
        modifies=$("${g[@]}" status --short)
    fi
    local avance retard
    avance=$("${g[@]}" rev-list --count origin/main..HEAD)
    retard=$("${g[@]}" rev-list --count HEAD..origin/main)

    echo "   en avance de $avance, en retard de $retard sur origin/main"
    if [ -n "$modifies" ]; then
        echo "   fichiers modifiés ou non suivis :"
        echo "$modifies" | sed 's/^/     /'
    else
        echo "   arbre propre"
    fi
    if [ "$retard" -gt 0 ]; then
        echo "   commits à récupérer :"
        "${g[@]}" log --format='     %h %s' HEAD..origin/main | cut -c1-150
    fi
    if [ "$avance" -gt 0 ]; then
        echo "   commits à pousser :"
        "${g[@]}" log --format='     %h %s' origin/main..HEAD | cut -c1-150
    fi

    [ "$mode" = "pull" ] || return
    if [ "$retard" -eq 0 ]; then
        echo "   à jour, pull sauté"
        return
    fi
    if [ "$avance" -gt 0 ]; then
        echo "   DIVERGENCE : pas de pull --ff-only possible, à résoudre à la main"
        probleme=1
        return
    fi
    local conflit
    conflit=$(comm -12 \
        <("${g[@]}" diff --name-only HEAD origin/main | sort) \
        <({
            "${g[@]}" diff --name-only
            "${g[@]}" diff --name-only --cached
        } | sort -u))
    if [ -n "$conflit" ]; then
        echo "   PULL NON FAIT : fichiers modifiés localement et touchés par le pull :"
        echo "$conflit" | sed 's/^/     /'
        probleme=1
        return
    fi
    if "${g[@]}" pull -q --ff-only origin main; then
        echo "   pull fait : $retard commit(s) récupéré(s)"
    else
        echo "   PULL REFUSÉ par git (voir le message ci-dessus)"
        probleme=1
    fi
}

un_depot public git
un_depot interne git --git-dir=.git-interne --work-tree=.

exit $probleme
